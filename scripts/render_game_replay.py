#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "imageio-ffmpeg==0.6.0",
#   "pillow==11.3.0",
# ]
# ///
"""Render an agario-kit ``output/game.json`` event log to an MP4 replay.

The renderer deliberately uses the full arena rather than a player's limited
vision.  It is intended for strategy review: player positions, every blob,
food, viruses, the live mass ranking, round number, and recent consumptions are
visible in one frame.

Example:
    uv run scripts/render_game_replay.py \
        .agario/simulation/output/game.json \
        --output .agario/simulation/output/replay.mp4 \
        --tracked-player 0
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


BACKGROUND = "#07131b"
PANEL = "#0c1c26"
GRID_MINOR = "#112a35"
GRID_MAJOR = "#1b3b47"
TEXT = "#e8f2f5"
MUTED = "#86a4af"
FOOD = "#ffd166"
VIRUS = "#58cf5f"
PLAYER_COLOURS = (
    "#5ee6a8",
    "#ff6b6b",
    "#5ca9ff",
    "#f8a84e",
    "#c77dff",
    "#4dd8df",
    "#ff82b2",
    "#d6e65e",
)


@dataclass
class PlayerState:
    player_id: int
    alive: bool
    blobs: list[dict[str, Any]] = field(default_factory=list)
    direction: tuple[float, float] = (0.0, 0.0)
    split_requested: bool = False

    @property
    def mass(self) -> float:
        # Engine mass is area-normalised, i.e. the sum of radius squared.
        return sum(float(blob["radius"]) ** 2 for blob in self.blobs)

    @property
    def centre(self) -> tuple[float, float] | None:
        if not self.blobs:
            return None
        total = sum(float(blob["radius"]) ** 2 for blob in self.blobs)
        if total <= 0:
            return None
        x = sum(float(blob["pos"][0]) * float(blob["radius"]) ** 2 for blob in self.blobs)
        y = sum(float(blob["pos"][1]) * float(blob["radius"]) ** 2 for blob in self.blobs)
        return (x / total, y / total)


@dataclass
class ReplayState:
    arena_size: float
    max_rounds: int
    turn_duration: float
    players: dict[int, PlayerState]
    food: dict[int, tuple[float, float]] = field(default_factory=dict)
    viruses: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    round_number: int = 0
    winner: int | None = None
    round_events: list[str] = field(default_factory=list)
    moved_this_round: set[int] = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an agario-kit game.json event log as an MP4 replay."
    )
    parser.add_argument("game_json", type=Path, help="Path to output/game.json")
    parser.add_argument(
        "--output",
        type=Path,
        help="MP4 destination (default: replay.mp4 beside game.json)",
    )
    parser.add_argument(
        "--tracked-player",
        type=int,
        default=0,
        help="Player to highlight and show movement intent for (default: 0)",
    )
    parser.add_argument("--fps", type=int, default=20, help="Output frames per second")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Render every Nth game round (default: 1)",
    )
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument(
        "--end-round",
        type=int,
        help="Last round to render, inclusive (default: full match)",
    )
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=2.0,
        help="Hold the final rendered frame for this many seconds",
    )
    args = parser.parse_args()
    if args.fps <= 0 or args.stride <= 0:
        parser.error("--fps and --stride must be positive")
    if args.start_round < 0:
        parser.error("--start-round must not be negative")
    if args.end_round is not None and args.end_round < args.start_round:
        parser.error("--end-round must be greater than or equal to --start-round")
    if args.width < 640 or args.height < 480:
        parser.error("frame size must be at least 640x480")
    if args.width % 2 or args.height % 2:
        parser.error("H.264 frame width and height must be even")
    return args


def load_events(path: Path) -> list[dict[str, Any]]:
    try:
        events = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read replay {path}: {exc}") from exc
    if not isinstance(events, list) or not events:
        raise SystemExit(f"Replay {path} must contain a non-empty JSON event array")
    if events[0].get("event_type") != "event_game_started":
        raise SystemExit(f"Replay {path} does not start with event_game_started")
    return events


def initial_state(start: dict[str, Any]) -> ReplayState:
    players: dict[int, PlayerState] = {}
    for raw in start["players"]:
        player_id = int(raw["player_id"])
        players[player_id] = PlayerState(
            player_id=player_id,
            alive=bool(raw.get("alive", True)),
            blobs=list(raw.get("blobs", [])),
        )
    return ReplayState(
        arena_size=float(start["arena_size"]),
        max_rounds=int(start["max_rounds"]),
        turn_duration=float(start.get("turn_duration_seconds", 0.1)),
        players=players,
    )


def apply_event(state: ReplayState, event: dict[str, Any]) -> bool:
    """Apply one event and return True after a complete game round."""
    event_type = event.get("event_type")
    if event_type == "event_food_spawned":
        for food in event.get("foods", []):
            state.food[int(food["food_id"])] = tuple(food["pos"])
    elif event_type == "event_virus_spawned":
        for virus in event.get("viruses", []):
            state.viruses[int(virus["virus_id"])] = (
                float(virus["pos"][0]),
                float(virus["pos"][1]),
                float(virus["radius"]),
            )
    elif event_type == "event_food_eaten":
        for food_id in event.get("food_ids", []):
            state.food.pop(int(food_id), None)
    elif event_type == "event_virus_consumed":
        state.viruses.pop(int(event["virus_id"]), None)
        state.round_events.append(
            f"P{event['player_id']} consumed virus V{event['virus_id']} "
            f"(+{event.get('pieces_created', '?')} cells)"
        )
    elif event_type == "event_player_eaten":
        eater = int(event["eater_player_id"])
        eaten = int(event["eaten_player_id"])
        state.round_events.append(f"P{eater} ate a P{eaten} cell")
        if not event.get("eaten_player_alive", True) and eaten in state.players:
            state.players[eaten].alive = False
    elif event_type == "move_player":
        player = state.players.get(int(event["player_id"]))
        if player is not None:
            direction = event.get("direction", {})
            player.direction = (float(direction.get("x", 0.0)), float(direction.get("y", 0.0)))
            player.split_requested = bool(event.get("split", False))
    elif event_type == "event_player_moved":
        player_id = int(event["player_id"])
        player = state.players[player_id]
        player.alive = bool(event.get("alive", True))
        player.blobs = list(event.get("blobs", []))
        # Every active slot produces exactly one moved event per engine round.
        state.moved_this_round.add(player_id)
        if len(state.moved_this_round) == len(state.players):
            state.round_number += 1
            state.moved_this_round.clear()
            return True
    elif event_type == "event_player_won":
        state.winner = int(event["player_id"])
    return False


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ("DejaVuSans-Bold.ttf", "Arial Bold.ttf")
        if bold
        else ("DejaVuSans.ttf", "Arial.ttf")
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def draw_virus(draw: ImageDraw.ImageDraw, x: float, y: float, radius: float) -> None:
    points: list[tuple[float, float]] = []
    for index in range(24):
        angle = math.pi * index / 12 - math.pi / 2
        length = radius if index % 2 == 0 else radius * 0.82
        points.append((x + math.cos(angle) * length, y + math.sin(angle) * length))
    draw.polygon(points, fill=VIRUS, outline="#d4ffd9", width=1)
    core = radius * 0.70
    draw.ellipse((x - core, y - core, x + core, y + core), fill="#43ba52")


def render_frame(
    state: ReplayState,
    *,
    width: int,
    height: int,
    tracked_player: int,
    final_ranking: list[int] | None,
    final_masses: dict[str, float] | None,
) -> Image.Image:
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    margin = 24
    footer_height = 48
    arena_px = min(height - margin * 2 - footer_height, width - 390)
    arena_left = margin
    arena_top = margin
    arena_right = arena_left + arena_px
    arena_bottom = arena_top + arena_px
    scale = arena_px / state.arena_size

    draw.rounded_rectangle(
        (arena_left - 8, arena_top - 8, arena_right + 8, arena_bottom + 8),
        radius=10,
        fill="#091820",
        outline="#31505c",
        width=2,
    )
    for world in range(0, math.ceil(state.arena_size) + 1, 2):
        screen = arena_left + world * scale
        colour = GRID_MAJOR if world % 10 == 0 else GRID_MINOR
        draw.line((screen, arena_top, screen, arena_bottom), fill=colour, width=1)
        screen_y = arena_top + world * scale
        draw.line((arena_left, screen_y, arena_right, screen_y), fill=colour, width=1)

    def screen(pos: Iterable[float]) -> tuple[float, float]:
        x, y = pos
        return arena_left + float(x) * scale, arena_top + float(y) * scale

    food_radius = max(1.3, scale * 0.13)
    for pos in state.food.values():
        x, y = screen(pos)
        draw.ellipse(
            (x - food_radius, y - food_radius, x + food_radius, y + food_radius),
            fill=FOOD,
        )
    for x_world, y_world, radius_world in state.viruses.values():
        x, y = screen((x_world, y_world))
        draw_virus(draw, x, y, max(5.0, radius_world * scale))

    small_font = load_font(11, bold=True)
    # Large cells are drawn first, so small cells and labels remain visible.
    blob_records: list[tuple[float, int, dict[str, Any]]] = []
    for player_id, player in state.players.items():
        for blob in player.blobs:
            blob_records.append((float(blob["radius"]), player_id, blob))
    labelled_players: set[int] = set()
    for radius_world, player_id, blob in sorted(
        blob_records, key=lambda record: (record[0], record[1]), reverse=True
    ):
        x, y = screen(blob["pos"])
        radius = max(3.0, radius_world * scale)
        colour = PLAYER_COLOURS[player_id % len(PLAYER_COLOURS)]
        tracked = player_id == tracked_player
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=colour,
            outline="#ffffff" if tracked else "#d7e5e9",
            width=3 if tracked else 1,
        )
        if tracked:
            draw.ellipse(
                (x - radius - 3, y - radius - 3, x + radius + 3, y + radius + 3),
                outline="#65f7b8",
                width=2,
            )
        if player_id not in labelled_players and radius >= 8:
            label = f"P{player_id}"
            box = draw.textbbox((0, 0), label, font=small_font)
            draw.text(
                (x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2 - 1),
                label,
                fill="#08131a",
                font=small_font,
                stroke_width=1 if radius >= 14 else 0,
                stroke_fill="#ffffff80",
            )
            labelled_players.add(player_id)

    tracked = state.players.get(tracked_player)
    if tracked is not None and tracked.centre is not None:
        cx, cy = screen(tracked.centre)
        dx, dy = tracked.direction
        length = math.hypot(dx, dy)
        if length > 1e-8:
            arrow_length = 30 if not tracked.split_requested else 42
            ex = cx + dx / length * arrow_length
            ey = cy + dy / length * arrow_length
            draw.line((cx, cy, ex, ey), fill="#ffffff", width=3)
            draw.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill="#ffffff")

    panel_left = arena_right + 28
    panel_right = width - margin
    draw.rounded_rectangle(
        (panel_left, margin - 8, panel_right, arena_bottom + 8),
        radius=10,
        fill=PANEL,
        outline="#24404c",
        width=2,
    )
    title_font = load_font(24, bold=True)
    heading_font = load_font(15, bold=True)
    body_font = load_font(14)
    tiny_font = load_font(12)
    draw.text((panel_left + 18, margin + 12), "MATCH REPLAY", fill=TEXT, font=title_font)
    draw.text(
        (panel_left + 18, margin + 52),
        f"Round {state.round_number:04d} / {state.max_rounds}",
        fill="#65f7b8",
        font=heading_font,
    )
    draw.text(
        (panel_left + 18, margin + 76),
        f"Food {len(state.food):3d}   Viruses {len(state.viruses):2d}",
        fill=MUTED,
        font=tiny_font,
    )
    draw.line((panel_left + 16, margin + 105, panel_right - 16, margin + 105), fill="#29424d")
    draw.text((panel_left + 18, margin + 120), "LIVE RANKING", fill=TEXT, font=heading_font)

    ranking = sorted(state.players, key=lambda pid: (-state.players[pid].mass, pid))
    if state.winner is not None and final_ranking:
        ranking = final_ranking
    row_y = margin + 151
    for place, player_id in enumerate(ranking, 1):
        player = state.players[player_id]
        colour = PLAYER_COLOURS[player_id % len(PLAYER_COLOURS)]
        if player_id == tracked_player:
            draw.rounded_rectangle(
                (panel_left + 10, row_y - 3, panel_right - 10, row_y + 24),
                radius=5,
                fill="#14352f",
            )
        draw.ellipse((panel_left + 18, row_y + 4, panel_left + 30, row_y + 16), fill=colour)
        draw.text((panel_left + 38, row_y), f"#{place}  P{player_id}", fill=TEXT, font=body_font)
        mass = player.mass
        if state.winner is not None and final_masses:
            mass = float(final_masses.get(str(player_id), mass))
        suffix = "  TRACK" if player_id == tracked_player else ""
        right_text = f"{mass:6.1f}  {len(player.blobs):2d}c{suffix}"
        box = draw.textbbox((0, 0), right_text, font=tiny_font)
        draw.text((panel_right - 18 - (box[2] - box[0]), row_y + 3), right_text, fill=MUTED, font=tiny_font)
        row_y += 31

    event_top = margin + 415
    draw.line((panel_left + 16, event_top - 14, panel_right - 16, event_top - 14), fill="#29424d")
    draw.text((panel_left + 18, event_top), "ROUND EVENTS", fill=TEXT, font=heading_font)
    event_y = event_top + 27
    for message in state.round_events[-5:]:
        if len(message) > 42:
            message = message[:39] + "..."
        draw.text((panel_left + 18, event_y), message, fill=MUTED, font=tiny_font)
        event_y += 20

    tracked_mass = tracked.mass if tracked is not None else 0.0
    tracked_cells = len(tracked.blobs) if tracked is not None else 0
    tracked_alive = tracked.alive if tracked is not None else False
    footer_y = arena_bottom + 21
    status = "ALIVE" if tracked_alive else "RESPAWNING"
    draw.text(
        (arena_left, footer_y),
        f"Tracked P{tracked_player}   {status}   mass {tracked_mass:.2f}   cells {tracked_cells}",
        fill=TEXT,
        font=body_font,
    )
    legend = "circle = player   star = virus   dot = food   arrow = tracked command"
    box = draw.textbbox((0, 0), legend, font=tiny_font)
    draw.text((arena_right - (box[2] - box[0]), footer_y + 2), legend, fill=MUTED, font=tiny_font)

    if state.winner is not None:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        banner = (arena_left + 90, arena_top + arena_px / 2 - 48, arena_right - 90, arena_top + arena_px / 2 + 48)
        overlay_draw.rounded_rectangle(banner, radius=14, fill=(4, 17, 24, 220), outline="#65f7b8", width=3)
        message = f"MATCH OVER  —  P{state.winner} WINS"
        box = overlay_draw.textbbox((0, 0), message, font=title_font)
        overlay_draw.text(
            ((banner[0] + banner[2] - (box[2] - box[0])) / 2, banner[1] + 31),
            message,
            fill=TEXT,
            font=title_font,
        )
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return image


def load_results(game_json: Path) -> tuple[list[int] | None, dict[str, float] | None]:
    results_path = game_json.with_name("results.json")
    if not results_path.exists():
        return None, None
    try:
        results = json.loads(results_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, None
    ranking = results.get("ranking")
    masses = results.get("final_masses")
    return (list(map(int, ranking)) if ranking else None, masses if isinstance(masses, dict) else None)


def encode(
    events: list[dict[str, Any]],
    *,
    destination: Path,
    fps: int,
    stride: int,
    start_round: int,
    end_round: int | None,
    width: int,
    height: int,
    tracked_player: int,
    hold_seconds: float,
    final_ranking: list[int] | None,
    final_masses: dict[str, float] | None,
) -> tuple[int, ReplayState]:
    state = initial_state(events[0])
    if tracked_player not in state.players:
        raise SystemExit(f"Tracked player {tracked_player} is not present in this replay")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    frames = 0
    last_frame: Image.Image | None = None

    def write_current_frame() -> None:
        nonlocal frames, last_frame
        last_frame = render_frame(
            state,
            width=width,
            height=height,
            tracked_player=tracked_player,
            final_ranking=final_ranking,
            final_masses=final_masses,
        )
        process.stdin.write(last_frame.tobytes())
        frames += 1

    try:
        if start_round == 0:
            write_current_frame()
        for event in events[1:]:
            round_complete = apply_event(state, event)
            if event.get("event_type") == "event_player_won":
                # The win event follows the last completed round; replace the
                # ordinary last image with an additional explicit end frame.
                write_current_frame()
            elif round_complete:
                selected = (
                    state.round_number >= start_round
                    and (end_round is None or state.round_number <= end_round)
                    and (state.round_number - start_round) % stride == 0
                )
                if selected:
                    write_current_frame()
                state.round_events.clear()
                for player in state.players.values():
                    player.split_requested = False
                if end_round is not None and state.round_number >= end_round:
                    break
        if last_frame is None:
            raise SystemExit("No frames selected; check --start-round/--end-round")
        for _ in range(max(0, round(hold_seconds * fps))):
            process.stdin.write(last_frame.tobytes())
            frames += 1
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            destination.unlink(missing_ok=True)
            raise SystemExit(f"ffmpeg failed with exit code {return_code}:\n{stderr[-4000:]}")
    except BaseException:
        process.kill()
        process.wait()
        destination.unlink(missing_ok=True)
        raise
    return frames, state


def main() -> None:
    args = parse_args()
    game_json = args.game_json.expanduser().resolve()
    destination = (args.output or game_json.with_name("replay.mp4")).expanduser().resolve()
    events = load_events(game_json)
    final_ranking, final_masses = load_results(game_json)
    frames, state = encode(
        events,
        destination=destination,
        fps=args.fps,
        stride=args.stride,
        start_round=args.start_round,
        end_round=args.end_round,
        width=args.width,
        height=args.height,
        tracked_player=args.tracked_player,
        hold_seconds=args.hold_seconds,
        final_ranking=final_ranking,
        final_masses=final_masses,
    )
    duration = frames / args.fps
    print(
        f"Rendered {frames} frames ({duration:.1f}s) through round "
        f"{state.round_number} to {destination}"
    )


if __name__ == "__main__":
    main()

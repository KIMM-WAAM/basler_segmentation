"""흰 배경에 그린 빨간 와이어를 인스턴스별 기준 OBB로 읽는다."""
from pathlib import Path

from edge import obb_pose
from PIL import Image, ImageChops

from preprocess import min_area_rect


def red_wire_templates(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with Image.open(path) as im:
        r, g, b, a = im.convert("RGBA").split()
        width, height = im.size
    red = ImageChops.multiply(
        ImageChops.subtract(r, g).point(lambda v: 255 if v > 60 else 0),
        ImageChops.subtract(r, b).point(lambda v: 255 if v > 60 else 0),
    )
    mask = ImageChops.multiply(red, a.point(lambda v: 255 if v else 0))
    remaining = {i for i, value in enumerate(mask.tobytes()) if value}
    components = []
    while remaining:
        start = remaining.pop()
        stack, points = [start], []
        while stack:
            index = stack.pop()
            x, y = index % width, index // width
            points.append((x, y))
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    neighbor = ny * width + nx
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        if len(points) >= 20:
            components.append(points)
    templates = []
    for points in components:
        rectangle, area = min_area_rect(points)
        if area > 0:
            templates.append({**obb_pose(rectangle), "size": [width, height]})
    return sorted(templates, key=lambda t: t["pivot"][0])

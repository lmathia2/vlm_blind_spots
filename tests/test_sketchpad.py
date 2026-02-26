"""Tests for sketchpad vision primitives and query decomposition."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from sketchpad import (
    PRIMITIVE_REGISTRY,
    DECOMPOSITION_TEMPLATES,
    QUERY_PATTERNS,
    FALLBACK_PLAN,
    classify_query,
    decompose_question,
    detect_edges,
    count_line_transitions,
    detect_contours,
    segment_colors,
    measure_bar_fill,
    detect_boxes,
    cluster_by_y,
    crop_and_enhance,
    detect_points,
    trace_colored_paths,
    run_sketchpad_pass0,
    build_sketchpad_prompt,
    parse_sketchpad_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def white_image():
    """A plain white 512x512 image."""
    return Image.new("RGB", (512, 512), color="white")


@pytest.fixture
def grid_image():
    """A simple 4x4 grid image with black lines on white background."""
    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)
    # Draw 5 vertical lines (creating 4 columns)
    for x in [0, 128, 256, 384, 511]:
        draw.line([(x, 0), (x, 511)], fill="black", width=2)
    # Draw 5 horizontal lines (creating 4 rows)
    for y in [0, 128, 256, 384, 511]:
        draw.line([(0, y), (511, y)], fill="black", width=2)
    return img


@pytest.fixture
def nested_squares_image():
    """An image with 3 nested squares."""
    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)
    # Outer square
    draw.rectangle([(50, 50), (462, 462)], outline="black", width=2)
    # Middle square
    draw.rectangle([(130, 130), (382, 382)], outline="black", width=2)
    # Inner square
    draw.rectangle([(200, 200), (312, 312)], outline="black", width=2)
    return img


@pytest.fixture
def colored_regions_image():
    """An image with distinct colored regions (simulating a pie chart)."""
    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)
    # Red region (top-left quadrant)
    draw.rectangle([(0, 0), (255, 255)], fill=(200, 30, 30))
    # Blue region (top-right quadrant)
    draw.rectangle([(256, 0), (511, 255)], fill=(30, 30, 200))
    # Green region (bottom-left quadrant)
    draw.rectangle([(0, 256), (255, 511)], fill=(30, 200, 30))
    return img


@pytest.fixture
def progress_bar_image():
    """An image with a progress bar (green fill on gray background)."""
    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)
    # Bar background (gray)
    draw.rectangle([(50, 200), (462, 240)], fill=(224, 224, 224))
    # Bar fill (green, ~50%)
    draw.rectangle([(50, 200), (256, 240)], fill=(76, 175, 80))
    return img


@pytest.fixture
def scatter_image():
    """An image with colored dots (simulating scatter points)."""
    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)
    # Blue dot at (100, 100)
    draw.ellipse([(95, 95), (115, 115)], fill=(31, 119, 180))
    # Red dot at (300, 200)
    draw.ellipse([(295, 195), (315, 215)], fill=(214, 39, 40))
    # Green dot at (400, 350)
    draw.ellipse([(395, 345), (415, 365)], fill=(44, 160, 44))
    return img


# ---------------------------------------------------------------------------
# Primitive Registry
# ---------------------------------------------------------------------------

class TestPrimitiveRegistry:
    def test_all_primitives_registered(self):
        expected = {
            "detect_edges", "count_line_transitions", "detect_contours",
            "segment_colors", "measure_bar_fill", "detect_boxes",
            "cluster_by_y", "crop_and_enhance", "trace_colored_paths",
            "detect_points",
        }
        assert expected == set(PRIMITIVE_REGISTRY.keys())

    def test_all_callables(self):
        for name, fn in PRIMITIVE_REGISTRY.items():
            assert callable(fn), f"{name} is not callable"


# ---------------------------------------------------------------------------
# Individual Primitive Tests
# ---------------------------------------------------------------------------

class TestDetectEdges:
    def test_returns_tuple(self, white_image):
        result, findings = detect_edges(white_image)
        assert isinstance(result, Image.Image)
        assert isinstance(findings, str)

    def test_detects_edges_in_grid(self, grid_image):
        result, findings = detect_edges(grid_image)
        assert "edge pixels" in findings.lower()
        assert result.size == grid_image.size


class TestCountLineTransitions:
    def test_horizontal_scan(self, grid_image):
        result, findings = count_line_transitions(grid_image, axis="horizontal")
        assert isinstance(result, Image.Image)
        assert "vertical line" in findings.lower()

    def test_vertical_scan(self, grid_image):
        result, findings = count_line_transitions(grid_image, axis="vertical")
        assert isinstance(result, Image.Image)
        assert "horizontal line" in findings.lower()

    def test_white_image_no_transitions(self, white_image):
        _, findings = count_line_transitions(white_image, axis="horizontal")
        assert "0" in findings or "detected 0" in findings.lower()


class TestDetectContours:
    def test_finds_contours_in_squares(self, nested_squares_image):
        result, findings = detect_contours(nested_squares_image, min_area=50)
        assert isinstance(result, Image.Image)
        assert "contour" in findings.lower()
        # Should find at least 1 contour
        assert "found 0" not in findings.lower()

    def test_min_area_filters(self, white_image):
        _, findings = detect_contours(white_image, min_area=10000)
        assert "0" in findings or "found 0" in findings.lower()


class TestSegmentColors:
    def test_segments_colored_regions(self, colored_regions_image):
        result, findings = segment_colors(colored_regions_image)
        assert isinstance(result, Image.Image)
        # Should detect red, blue, green regions
        assert "%" in findings  # percentages should be present

    def test_white_image_excluded(self, white_image):
        _, findings = segment_colors(white_image, exclude_white=True)
        assert isinstance(findings, str)


class TestMeasureBarFill:
    def test_measures_progress_bar(self, progress_bar_image):
        result, findings = measure_bar_fill(progress_bar_image)
        assert isinstance(result, Image.Image)
        assert "bar" in findings.lower()
        # Should detect approximately 50% fill
        if "%" in findings:
            import re
            pcts = re.findall(r"(\d+\.?\d*)%", findings)
            if pcts:
                pct = float(pcts[0])
                assert 30 <= pct <= 70, f"Expected ~50%, got {pct}%"


class TestDetectBoxes:
    def test_finds_boxes(self, nested_squares_image):
        result, findings = detect_boxes(nested_squares_image)
        assert isinstance(result, Image.Image)
        assert "box" in findings.lower() or "rectangular" in findings.lower()

    def test_returns_image(self, white_image):
        result, findings = detect_boxes(white_image)
        assert isinstance(result, Image.Image)


class TestClusterByY:
    def test_clusters_boxes(self, white_image):
        # Create image with boxes at two y-levels
        img = Image.new("RGB", (512, 512), color="white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([(50, 50), (150, 100)], outline="black", width=2)
        draw.rectangle([(200, 50), (300, 100)], outline="black", width=2)
        draw.rectangle([(100, 300), (250, 350)], outline="black", width=2)

        result, findings = cluster_by_y(img)
        assert isinstance(result, Image.Image)
        assert "level" in findings.lower()

    def test_no_boxes(self, white_image):
        result, findings = cluster_by_y(white_image, boxes=[])
        assert "no boxes" in findings.lower()


class TestCropAndEnhance:
    def test_center_crop(self, grid_image):
        result, findings = crop_and_enhance(grid_image, region="center")
        assert isinstance(result, Image.Image)
        assert "center" in findings

    def test_full_enhance(self, grid_image):
        result, findings = crop_and_enhance(grid_image, region="full")
        assert isinstance(result, Image.Image)
        assert "full" in findings

    def test_all_regions(self, grid_image):
        regions = [
            "center", "top-left", "top-right", "bottom-left",
            "bottom-right", "top", "bottom", "left", "right", "full",
        ]
        for region in regions:
            result, findings = crop_and_enhance(grid_image, region=region)
            assert isinstance(result, Image.Image)


class TestDetectPoints:
    def test_finds_colored_points(self, scatter_image):
        result, findings = detect_points(scatter_image)
        assert isinstance(result, Image.Image)
        assert "point" in findings.lower()

    def test_white_image_no_points(self, white_image):
        _, findings = detect_points(white_image)
        assert "0" in findings


class TestTraceColoredPaths:
    def test_traces_colored_lines(self):
        """Create image with colored lines and verify detection."""
        img = Image.new("RGB", (512, 512), color="white")
        draw = ImageDraw.Draw(img)
        # Red line
        draw.line([(50, 100), (462, 100)], fill=(214, 39, 40), width=6)
        # Blue line
        draw.line([(50, 300), (462, 300)], fill=(31, 119, 180), width=6)

        result, findings = trace_colored_paths(img)
        assert isinstance(result, Image.Image)
        assert "path" in findings.lower()

    def test_no_colored_paths(self, white_image):
        _, findings = trace_colored_paths(white_image)
        assert "0" in findings


# ---------------------------------------------------------------------------
# Question Decomposition
# ---------------------------------------------------------------------------

class TestDecomposeQuestion:
    def test_known_task_uses_template(self):
        subs = decompose_question("any prompt", task_name="pie_chart")
        assert len(subs) == 2
        assert "color" in subs[0].lower()

    def test_counting_grid_decomposition(self):
        subs = decompose_question("How many rows and columns?", task_name="counting_grid")
        assert len(subs) == 2
        assert "horizontal" in subs[0].lower()
        assert "vertical" in subs[1].lower()

    def test_hierarchy_decomposition(self):
        subs = decompose_question("How deep?", task_name="hierarchy_depth")
        assert len(subs) == 2

    def test_unknown_task_passthrough(self):
        subs = decompose_question("What is in this image?", task_name="unknown_task")
        assert len(subs) == 1
        assert subs[0] == "What is in this image?"

    def test_and_splitting(self):
        subs = decompose_question(
            "What color is the top shape and how many sides does it have?",
            task_name="unknown",
        )
        assert len(subs) == 2

    def test_simple_question_no_split(self):
        subs = decompose_question("How many squares?", task_name="")
        assert len(subs) == 1

    def test_all_templates_exist(self):
        for task_name in DECOMPOSITION_TEMPLATES:
            subs = decompose_question("dummy", task_name=task_name)
            assert len(subs) >= 1


# ---------------------------------------------------------------------------
# Query Classification
# ---------------------------------------------------------------------------

class TestClassifyQuery:
    def test_grid_query(self):
        plan = classify_query("How many rows and columns in the grid?")
        prim_names = [p[0] for p in plan]
        assert "count_line_transitions" in prim_names

    def test_counting_shapes(self):
        plan = classify_query("Count the number of squares")
        prim_names = [p[0] for p in plan]
        assert "detect_contours" in prim_names

    def test_hierarchy(self):
        plan = classify_query("How many levels deep is this hierarchy?")
        prim_names = [p[0] for p in plan]
        assert "detect_boxes" in prim_names

    def test_proportion(self):
        plan = classify_query("What percentage does the blue slice represent?")
        prim_names = [p[0] for p in plan]
        assert "segment_colors" in prim_names

    def test_path_tracing(self):
        plan = classify_query("How many paths go from A to B?")
        prim_names = [p[0] for p in plan]
        assert "trace_colored_paths" in prim_names

    def test_text_reading(self):
        plan = classify_query("What does the text say?")
        prim_names = [p[0] for p in plan]
        assert "crop_and_enhance" in prim_names

    def test_fallback(self):
        plan = classify_query("xyzzy123")
        assert plan == FALLBACK_PLAN

    def test_task_name_override(self):
        plan = classify_query("some ambiguous question", task_name="pie_chart")
        prim_names = [p[0] for p in plan]
        assert "segment_colors" in prim_names

    def test_scatter(self):
        plan = classify_query("What is the y-value at x=5?")
        prim_names = [p[0] for p in plan]
        assert "detect_points" in prim_names


# ---------------------------------------------------------------------------
# Sketchpad Orchestrator
# ---------------------------------------------------------------------------

class TestRunSketchpadPass0:
    def test_returns_annotated_image_and_findings(self, white_image):
        canvas, findings = run_sketchpad_pass0(white_image, "How many squares?", "nested_squares")
        assert isinstance(canvas, Image.Image)
        assert isinstance(findings, list)
        assert len(findings) >= 1
        assert "sub_question" in findings[0]
        assert "findings" in findings[0]
        assert "primitives_run" in findings[0]

    def test_hierarchy_runs_two_primitives(self, white_image):
        canvas, findings = run_sketchpad_pass0(
            white_image, "How deep?", "hierarchy_depth"
        )
        all_prims = []
        for f in findings:
            all_prims.extend(f["primitives_run"])
        assert len(all_prims) >= 2  # detect_boxes + cluster_by_y


class TestBuildSketchpadPrompt:
    def test_includes_findings(self):
        findings = [{
            "sub_question": "How many colors?",
            "primitives_run": ["segment_colors"],
            "findings": "Found 3 colors",
        }]
        prompt = build_sketchpad_prompt("What color?", findings, pass_num=1)
        assert "How many colors?" in prompt
        assert "Found 3 colors" in prompt
        assert "TOOL" in prompt
        assert "ANSWER" in prompt

    def test_includes_all_tools(self):
        prompt = build_sketchpad_prompt("Q?", [], pass_num=1)
        assert "crop_and_enhance" in prompt
        assert "detect_contours" in prompt
        assert "segment_colors" in prompt


class TestParseSketchpadResponse:
    def test_parse_tool_no_args(self):
        action, value, kwargs = parse_sketchpad_response("TOOL(detect_edges)")
        assert action == "tool"
        assert value == "detect_edges"

    def test_parse_tool_with_args(self):
        action, value, kwargs = parse_sketchpad_response(
            "TOOL(crop_and_enhance, center)"
        )
        assert action == "tool"
        assert value == "crop_and_enhance"
        assert kwargs["region"] == "center"

    def test_parse_tool_with_kv_args(self):
        action, value, kwargs = parse_sketchpad_response(
            "TOOL(count_line_transitions, axis=horizontal)"
        )
        assert action == "tool"
        assert value == "count_line_transitions"
        assert kwargs["axis"] == "horizontal"

    def test_parse_answer(self):
        action, value, kwargs = parse_sketchpad_response("ANSWER {5}")
        assert action == "answer"
        assert "{5}" in value

    def test_parse_answer_with_text(self):
        action, value, kwargs = parse_sketchpad_response(
            "Based on analysis, ANSWER (B) 35%"
        )
        assert action == "answer"
        assert "(B)" in value

    def test_parse_unknown(self):
        action, value, kwargs = parse_sketchpad_response("I think the answer is 5")
        assert action == "unknown"
        assert "5" in value

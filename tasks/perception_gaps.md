# Gap Analysis: VLM Blind Spots — Prioritized Task Expansion

## Business Document Perceptual Workflows Analyzed

- **Reading a dashboard screenshot:** Requires discriminating line styles (solid vs dashed), reading tiny sparklines, identifying icon states (green check vs red X), reading text on colored panel backgrounds, and comparing bar heights side-by-side.
- **Processing a spreadsheet screenshot:** Requires noticing which cells are highlighted (conditional formatting), detecting bold vs regular text, distinguishing similar numbers ("1,456" vs "1,465"), and reading text on colored cell backgrounds.
- **Reviewing annotated documents:** Requires color discrimination within annotation layers — multiple colors of annotation (red circle, blue arrow, yellow highlight).
- **Reading a multi-series chart:** Requires discriminating between line styles (solid/dashed/dotted), reading marker shapes (circle/square/triangle), and counting how many distinct series there are when lines overlap.

---

## Tier A — Large perceptual gaps, high business frequency

| # | Task | Perceptual Skill Tested | Why It's Missing | Business Context |
|---|------|------------------------|-----------------|-----------------|
| 1 | **Scatter plot value** (MC4) | Reading point position from an unconnected cloud | No task tests spatial position without connected paths or grid lines | Analytics dashboards, data exploration |
| 2 | **Heatmap cell value** (MC4) | Mapping continuous color intensity to numeric value | No task tests color gradient → value. All current P7 tests are categorical (which color = which series) | Correlation matrices, performance grids, risk dashboards |
| 3 | **Line style discrimination** | Distinguishing solid / dashed / dotted lines | P5 (fine state discrimination) is only tested on geometric shapes and checkboxes. Line style is a different visual encoding entirely | Multi-series charts where color isn't the only differentiator |
| 4 | **Highlighted text detection** | Detecting which words have a yellow/colored background highlight | P5 for text formatting — none of our tasks test formatting-level state changes in text | Document review, contract markup, tracked changes |
| 5 | **Color-coded table cells** | Identifying cell background colors in a grid | P7 is never tested in a tabular/grid context. All color tasks are in charts | Conditional formatting in spreadsheets — universal in business |
| 6 | **Grouped bar chart** (MC4) | Comparing adjacent bar heights across a gap | Perceptually different from stacked bars. Requires matching color → legend then comparing heights across groups | The most common chart type in business reports |

## Tier B — Meaningful gaps, common in practice

| # | Task | Perceptual Skill Tested | Why It's Missing | Business Context |
|---|------|------------------------|-----------------|-----------------|
| 7 | **Icon / status indicator** | Recognizing shape+color symbols (check, X, warning triangle, info circle) | P6 (symbol recognition) is severely undertested — only arrow direction. Icon recognition is a distinct perceptual act | Dashboard status panels, monitoring tools, notifications |
| 8 | **Text on colored background** | Reading text overlaid on non-white backgrounds | P8 is only tested on white backgrounds. Colored/gradient backgrounds are a major real-world degradation factor | Presentation slides, dashboard panels, banners, callout boxes |
| 9 | **Sparkline trend direction** | Perceiving up/down/flat trend from a tiny inline chart | P9 at extremely small scale (~30px tall). Nothing tests proportion judgment at this size | Excel sparklines, dashboard KPI cards, inline data viz |
| 10 | **Small digit discrimination** | Distinguishing "1,456" from "1,465" in adjacent cells | P8 precision — current text tasks test readability but not precision under confusable alternatives | Data entry validation, financial reconciliation, audit |
| 11 | **Relative comparison** | "Which bar is taller, A or B?" (no scale reading) | Pure P9 without any P8 involvement. Tests whether the model can do visual comparison independently of value reading | Quick visual comparison in any chart context |
| 12 | **Waterfall chart** (MC4) | Reading segment value from a non-zero baseline | P9 variant not covered — all current value-reading tasks have zero baselines except stacked bar | Financial bridges, P&L walks, variance analysis |

## Tier C — Valuable but more niche

| # | Task | Perceptual Skill Tested | Why It's Missing | Business Context |
|---|------|------------------------|-----------------|-----------------|
| 13 | **Marker shape discrimination** | Circle vs square vs triangle vs diamond markers | P6 for chart-specific symbols. Currently untested | Multi-series scatter/line charts |
| 14 | **Gauge / dial reading** (MC4) | Reading angular needle position on a semicircular scale | P9 angular reading different from pie (area) and bar (height) | Dashboard gauges, speedometers, KPI dials |
| 15 | **Multi-column reading order** | Reading text in correct column flow | P1 (spatial reference) for document layout. Currently only tested in grids/tables, not freeform text | PDF reports, newspaper layouts, multi-column forms |
| 16 | **Annotation color discrimination** | "Which word has the RED circle?" when there are red, blue, and green annotations | P7 in annotation context with multiple competing colors | Multi-reviewer document markup |
| 17 | **Series counting** | "How many distinct lines are in this chart?" | P3 (counting) + P2 (path following) when lines cross and overlap | Chart complexity assessment, data viz review |
| 18 | **Venn diagram membership** | Identifying items in overlap regions | P1 (spatial containment) for non-rectangular regions | Business presentations, market analysis |
| 19 | **Toggle / switch states** | On/off slider switch position | P5 for a different visual encoding than checkbox/radio | UI screenshots, settings panels |
| 20 | **Bold / font weight detection** | "Which words are bold?" | P5 for typographic weight discrimination | Document structure, emphasis detection |

---

## Primitive Coverage Analysis After Additions

| Primitive | Current Tasks | Gap Filled By |
|-----------|--------------|---------------|
| P5 (Fine State) | geometric shapes, checkboxes, radio, strikethrough | +line style, +highlight, +toggle, +bold |
| P6 (Symbol Recognition) | arrow direction only | +icons, +marker shapes |
| P7 (Color Discrimination) | legend matching, chart colors | +heatmap gradient, +conditional formatting, +annotation colors |
| P8 (Text Reading) | degraded, rotated, dense text on white | +colored backgrounds, +digit precision |
| P9 (Scale/Proportion) | bar value, line point, pie, progress bar | +scatter, +sparkline, +relative comparison, +gauge, +waterfall |

---

## Implementation Priority

**Tier A (this sprint):** scatter_plot, heatmap, line_style, highlighted_text, color_coded_cells, grouped_bar

**Tier B (next sprint):** icon_indicator, text_on_background, sparkline, digit_discrimination, relative_comparison, waterfall_chart

**Tier C (backlog):** marker_shape, gauge_dial, multi_column_order, annotation_color, series_counting, venn_diagram, toggle_switch, bold_detection

# Design: "Perimeter", an edge-device laboratory

The question the console answers is **what crosses the device's perimeter, and what it costs to keep it in.** The interface is organised around that boundary: the device on one side, the cloud on the other, and the payload drawn crossing between them.

## Identity

- **Surfaces:** graphite in dark mode (#141517 page, #1b1c1f panels), warm paper in light mode (#f6f6f4 page, #ffffff panels). Hairline rules, no shadows.
- **One accent:** perimeter violet (light #4a3aa7, dark #9085e9). It marks selection, focus and the device perimeter line, and nothing else. It was moved off teal so it cannot be confused with the aqua series colour.
- **Type:** Geist for text, Geist Mono for numbers, byte counts and identifiers. Numbers are tabular.
- **No** gradients, glass, neon or emoji.
- **Logo:** a square (the device) with a dashed edge and one short tick crossing it.

## Views

1. **Lab** (home). Choose a subject's recorded session, an architecture and a level. Three lanes:
   - **Sensor:** the replayed accelerometer and gyroscope traces, with the current window outlined.
   - **Perimeter:** what left the device for each window: its size in bytes, its kind (raw, features, label, histogram) and a compact rendering of its content.
   - **Cloud:** the decision for each window against the true activity, with mistakes marked.
   A transport bar steps through windows (space to play or pause, ←/→ to step).
2. **Frontier.** Accuracy against exposure for all 15 cells, with intervals; the Pareto set joined by a line. A segmented control switches the cost dimension shown as a second chart (bandwidth, device energy proxy, latency). Every chart has a table view.
3. **Grid.** The full architecture × level table: accuracy, macro-F1, utility loss, exposure, bytes per window, operations, latency and energy proxy, each with its interval.
4. **Experiments.** E2 (adversary strength), E3 (ε sweep), E4 (cost and break-even), E5 (streaming consistency): each with its question, figure, table and provenance line.
5. **Method.** The data, the three architectures, the five levels, the adversary, the cost model and its assumptions, and the limits.

## Data visualisation

- Architectures are the only categorical series (three), in a fixed order: centralized blue (#2a78d6 / dark #3987e5), edge orange (#eb6834 / #d95926), fully local aqua (#1baf7a / #199e70). The validator passes all checks in both themes on all pairs; light-mode aqua is below 3:1 against the panel, so every chart carries direct labels and a table view. Levels are shown by position and label (L0–L4), never by a fourth colour.
- Intervals are drawn as lines through the dot; a single y-axis per chart; direct labels on the three series, plus a legend.
- Status colours (ok, attention, bad) are reserved for correct/incorrect decisions in the Lab and always come with a glyph.

## States

Loading (skeleton rows), empty (no results yet: shows the command that produces them), error (API unreachable, with retry), and pending (an experiment not run yet). No chart is drawn without measured data.

## Responsive behaviour

Below 760 px the Lab lanes stack vertically and the frontier charts become full width; tables scroll horizontally inside a focusable region.

## Accessibility

WCAG 2.1 AA in both themes, checked with axe-core in browser tests. Every chart has a text table. Keyboard: all controls reachable; the Lab transport has shortcuts that are announced and do nothing while typing in a field. Motion is disabled under `prefers-reduced-motion`.

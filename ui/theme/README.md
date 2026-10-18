# Preserved GUI theme

The only part of the previous repository that survives Phase 1: the visual
language of the old control room, rebuilt as a green / black / gray design
system. There is no application code here.

| File         | Purpose                                                              |
| ------------ | -------------------------------------------------------------------- |
| `tokens.css` | Colour, type, spacing, radius, shadow and motion custom properties.   |
| `theme.css`  | Framework-agnostic components: shell, nav, top bar, cards, tables…    |
| `icons.svg`  | Icon sprite inherited from the original icon language (24×24 stroke). |
| `mark.svg`   | Recoloured logo mark (green gradient, dark glyph).                    |
| `fonts.css`  | Documents the local-only system font stack; ships no font binaries.  |
| `index.html` | Static reference page. Not connected to any backend.               |

## Preview

From `ui/theme`:

```bash
cd ui/theme && python3 -m http.server 8000
# open http://localhost:8000/
```

No build step, no npm, no CDN, no network calls. `index.html` is a design
reference: every value shown in it is a placeholder, not a measurement.

## What changed from the original theme

- Palette moved from pure black/white to green (`#55e38b` accent), black and
  gray surfaces; gradients are used only for depth and primary actions.
- Status colours now carry meaning (success / warning / danger / info); the
  original theme rendered all states in grayscale.
- Hard-coded gray literals were replaced by tokens.
- Layout, spacing rhythm, typography scale, icon set, shell structure and the
  compact data-card aesthetic were kept.

The theme is intentionally decoupled from any framework. A future frontend can
consume it via plain CSS, or the tokens can be ported to Tailwind/CSS-in-JS.

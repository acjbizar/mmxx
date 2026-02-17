# MMXX

Constructivist typeface.

## Letter Sample

![](src/sheet.svg)

## Web Fonts

```css
@font-face {
  font-family: "MMXX";
  src:
    url("https://hetcdn.nl/fonts/mmxx.woff2") format("woff2"),
    url("https://hetcdn.nl/fonts/mmxx.woff") format("woff"),
    url("https://hetcdn.nl/fonts/mmxx.ttf") format("truetype");
  font-weight: 400;
  font-style: normal;
  font-display: swap;
}

:root {
    --font-mmxx: "MMXX", system-ui, sans-serif;
}

.font--mmxx {
    font-family: var(--font-mmxx);
    font-variant-ligatures: none;
}
```

### CDN

```html
<link rel="stylesheet" href="https://hetcdn.nl/fonts/mmxx.css">
```

## Logo

![Logo](src/logo.svg)

## Tools

### Generate Video

```bash
# default: NO animation (static video)
python tools/generate-video.py --char=a

# animate toward white (default)
python tools/generate-video.py --char=a --to

# animate toward yellow
python tools/generate-video.py --char=a --to=yellow

# apply clouds theme
python tools/generate-video.py --chars=test --theme=clouds

# apply a theme only to the 2nd character (e in "test")
python tools/generate-video.py --chars=test --theme=matrix --only=2

# apply only to multiple characters (2nd and 4th)
python tools/generate-video.py --chars=test --theme=valentines --only=2,4

# Skip one glyph (keep original for glyph 2)
python tools/generate-video.py --chars=test --theme=classic --colors red - blue pink

# Export first frame as PNG
python tools/generate-video.py --chars=test --theme=clouds --png=0
```

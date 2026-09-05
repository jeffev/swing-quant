"""Turn the published article files into a standalone site under `site/`, for GitHub Pages.

The article files are written for the Artifact host, which wraps them in its own
`<!doctype html><head>…</head><body>` and supplies a small reset. Served directly they would be
a fragment: no doctype, no charset, no viewport, no language, and a `<title>` stranded in the
body. This script supplies exactly what the host used to and nothing more, so the page renders
the same in both places and there is only ever one copy of the article to edit.

It also rewires the two editions to each other. In the Artifacts each links to the other's URL;
on the site they are `/` and `/pt/`, and a relative link is the only version that keeps working
when the site moves.

The output is `site/`, not `docs/`: `docs/` already holds the project's planning notes and ADRs,
and serving that folder would publish them alongside the article. `site/` is derived — it is
gitignored on the working branch and pushed to `gh-pages`, which therefore contains the article
and nothing else.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARTICLES = REPO / "notebooks" / "article"
SITE = REPO / "site"

# The artifact host's own reset, reproduced. `[hidden]` is the load-bearing one: every
# chart/table toggle in the article works by setting the attribute, and a stylesheet that gave
# `.exhibit-body` a display would silently break all seven of them.
RESET = """<style>
  :root { color-scheme: light dark; }
  body { margin: 0; padding: 0; }
  img { max-width: 100%; }
  [hidden] { display: none !important; }
</style>"""


@dataclass(frozen=True)
class Edition:
    src: Path
    out: Path
    lang: str
    description: str
    other_url: str  # the artifact URL of the sibling edition, to be replaced
    other_href: str  # what it becomes on the site


EDITIONS = (
    Edition(
        src=ARTICLES / "swing_vs_investing.html",
        out=SITE / "index.html",
        lang="en",
        description=(
            "Sixteen years of daily bars across Brazil and the US: a validated swing-trading "
            "system against every passive alternative a household could actually buy — and what "
            "was really driving the returns."
        ),
        other_url="https://claude.ai/code/artifact/368e5a55-f206-440c-aad1-a9206a81349b",
        other_href="pt/",
    ),
    Edition(
        src=ARTICLES / "swing_vs_investing_pt.html",
        out=SITE / "pt" / "index.html",
        lang="pt-BR",
        description=(
            "Dezesseis anos de barras diárias no Brasil e nos EUA: um sistema de swing trade "
            "validado contra todas as alternativas passivas que uma família podia comprar — e o "
            "que estava de fato produzindo o retorno."
        ),
        other_url="https://claude.ai/code/artifact/06c14798-597d-4654-8445-babc58f3b643",
        other_href="../",
    ),
)

FAVICON = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E"
    "%3Ctext y='.9em' font-size='90'%3E%F0%9F%93%8A%3C/text%3E%3C/svg%3E"
)

TITLE_RE = re.compile(r"[ \t]*<title>(.*?)</title>\s*\n", re.S)


def build(edition: Edition, site_url: str | None = None) -> str:
    """Wrap one article as a complete HTML document. Returns the document."""
    body = edition.src.read_text(encoding="utf-8")

    match = TITLE_RE.search(body)
    if match is None:
        raise ValueError(f"{edition.src.name}: sem <title> para promover ao <head>")
    title = match.group(1).strip()
    body = TITLE_RE.sub("", body, count=1)

    if edition.other_url not in body:
        print(f"  aviso: {edition.src.name} não tem o link para a outra edição", file=sys.stderr)
    body = body.replace(edition.other_url, edition.other_href)

    canonical = ""
    social = ""
    if site_url:
        page = site_url.rstrip("/") + ("/" if edition.out.parent == SITE else "/pt/")
        canonical = f'\n<link rel="canonical" href="{page}">'
        social = (
            f'\n<meta property="og:type" content="article">'
            f'\n<meta property="og:title" content="{title}">'
            f'\n<meta property="og:description" content="{edition.description}">'
            f'\n<meta property="og:url" content="{page}">'
            f'\n<meta name="twitter:card" content="summary_large_image">'
        )

    return f"""<!doctype html>
<html lang="{edition.lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{edition.description}">
<meta name="author" content="jeffev">
<link rel="icon" href="{FAVICON}">{canonical}{social}
{RESET}
</head>
<body>
{body}</body>
</html>
"""


def main(site_url: str | None = None) -> None:
    SITE.mkdir(exist_ok=True)
    # Pages runs Jekyll by default, which would try to process the site and ignore anything it
    # does not understand. This file turns that off; the page is already built.
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    for edition in EDITIONS:
        if not edition.src.exists():
            print(f"{edition.src.name}: ausente, pulado")
            continue
        edition.out.parent.mkdir(parents=True, exist_ok=True)
        html = build(edition, site_url)
        edition.out.write_text(html, encoding="utf-8")
        print(f"{edition.out.relative_to(REPO)}: {len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "https://jeffev.github.io/swing-quant")

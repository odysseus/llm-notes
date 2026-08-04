# LLM Notes

A Docusaurus research library designed for GitHub Pages. Articles live in `docs/` as ordinary Markdown files. The site includes generated navigation, local full-text search, math rendering, light/dark themes, and a GitHub Issue form for updates and clarification requests.

## Run locally

Requires Node.js 20 or newer.

```bash
npm install
npm start
```

## Publish with GitHub Pages

1. Push the project to `odysseus/llm-notes` on GitHub.
2. In **Settings → Pages**, set **Source** to **GitHub Actions**.
3. Commit to the `main` branch. The included workflow builds and publishes the site.

The site is configured for `https://odysseus.github.io/llm-notes/`. During deployment it also reads GitHub's `GITHUB_REPOSITORY` environment variable so repository links remain consistent with the workflow context.

## Add or revise an article

- Add or edit a Markdown file in `docs/`.
- Add its document ID to `sidebars.js` if it should appear in the navigation.
- Push the change to `main`; GitHub Actions republishes the site.

The **Edit this page** link opens the current Markdown file on GitHub. The **Suggest an update** link opens the repository's structured research-note issue form.

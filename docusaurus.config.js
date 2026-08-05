const remarkMath = require('remark-math').default;
const rehypeKatex = require('rehype-katex').default;

const repository = process.env.GITHUB_REPOSITORY || 'odysseus/llm-notes';
const [organizationName, projectName] = repository.split('/');
const isUserSite = projectName.toLowerCase() === `${organizationName.toLowerCase()}.github.io`;
const baseUrl = process.env.DOCUSAURUS_BASE_URL || (isUserSite ? '/' : `/${projectName}/`);

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'LLM Notes',
  tagline: 'Concepts, papers, and practical architecture for building with language models',
  favicon: 'img/favicon.svg',
  url: `https://${organizationName}.github.io`,
  baseUrl,
  organizationName,
  projectName,
  trailingSlash: true,
  onBrokenLinks: 'throw',
  markdown: {
    format: 'detect',
    mermaid: true,
    hooks: {onBrokenMarkdownLinks: 'throw'},
  },
  themes: ['@docusaurus/theme-mermaid'],
  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: require.resolve('./sidebars.js'),
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          editUrl: `https://github.com/${organizationName}/${projectName}/edit/main/`,
          showLastUpdateTime: true,
        },
        blog: false,
        pages: false,
        theme: {customCss: require.resolve('./src/css/custom.css')},
        sitemap: {changefreq: 'weekly', priority: 0.6},
      },
    ],
  ],
  plugins: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        indexDocs: true,
        indexBlog: false,
        docsRouteBasePath: '/',
        language: ['en'],
        highlightSearchTermsOnTargetPage: true,
      },
    ],
  ],
  themeConfig: {
    colorMode: {defaultMode: 'light', respectPrefersColorScheme: true},
    navbar: {
      title: 'LLM Notes',
      items: [
        {type: 'docSidebar', sidebarId: 'researchSidebar', position: 'left', label: 'Library'},
        {
          href: `https://github.com/${organizationName}/${projectName}/issues/new?template=research-note.yml`,
          label: 'Suggest an update',
          position: 'right',
        },
        {
          href: `https://github.com/${organizationName}/${projectName}`,
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      copyright: 'LLM Notes · Built with Docusaurus',
    },
    docs: {sidebar: {hideable: true, autoCollapseCategories: false}},
    prism: {additionalLanguages: ['bash', 'json', 'rust']},
  },
};

module.exports = config;

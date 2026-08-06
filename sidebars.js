/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
module.exports = {
  researchSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Concepts and Architecture',
      collapsed: false,
      items: [
        'word-vectors',
        'retrieval-augmented-generation',
      ],
    },
    {
      type: 'category',
      label: 'Architecture',
      collapsed: false,
      items: [
        'chatbot-evolution',
        'llm-request-lifecycle',
        'context-engineering',
        'harness-engineering',
        'multi-agent-systems',
        'tool-use',
      ],
    },
    {
      type: 'category',
      label: 'Influential Papers',
      collapsed: false,
      items: [
        'paper-attention-is-all-you-need',
      ],
    },
  ],
};

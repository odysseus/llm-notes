/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
module.exports = {
  researchSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Concepts and Architecture',
      collapsed: false,
      items: [
        'llm-request-lifecycle',
        'context-engineering',
        'retrieval-augmented-generation',
      ],
    },
    {
      type: 'category',
      label: 'Papers and History',
      collapsed: false,
      items: [
        'paper-attention-is-all-you-need',
        'chatbot-evolution',
      ],
    },
  ],
};

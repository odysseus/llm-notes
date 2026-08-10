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
        'chatbot-evolution',
        'llm-request-lifecycle',
        'prompt-engineering',
        'retrieval-augmented-generation',
        'llm-security',
      ],
    },
    {
      type: 'category',
      label: 'Architecture',
      collapsed: false,
      items: [
        'instruction-tuning',
        'context-engineering',
        'tool-use',
        'multi-agent-systems',
        'harness-engineering',
        'llms-as-state-machines',
      ],
    },
    {
      type: 'category',
      label: 'Influential Papers',
      collapsed: false,
      items: [
        'paper-attention-is-all-you-need',
        'paper-toolformer',
        'paper-react',
      ],
    },
  ],
};

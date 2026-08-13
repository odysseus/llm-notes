/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
module.exports = {
  researchSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Concepts',
      collapsed: false,
      items: [
        'word-vectors',
        'chatbot-evolution',
        'llm-request-lifecycle',
        'prompt-engineering',
        'retrieval-augmented-generation',
        'instruction-tuning',
        'context-engineering',
        'tool-use',
        'task-specialization',
        'multi-turn-dialogue',
        'multi-agent-systems',
        'harness-engineering',
        'llms-as-state-machines',
        'llm-security',
      ],
    },
    {
      type: 'category',
      label: 'Briefs',
      collapsed: false,
      items: [
        'in-context-learning',
        'structured-output',
        'dynamic-selection',
        'soft-prompting',
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
        'paper-t5-text-to-text-transformer',
      ],
    },
  ],
};

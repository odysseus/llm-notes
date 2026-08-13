import React, {useEffect, useState} from 'react';

export default function Root({children}) {
  const [openPane, setOpenPane] = useState(null);

  useEffect(() => {
    const root = document.documentElement;

    root.classList.toggle('tablet-left-pane-open', openPane === 'left');
    root.classList.toggle('tablet-right-pane-open', openPane === 'right');

    return () => {
      root.classList.remove('tablet-left-pane-open', 'tablet-right-pane-open');
    };
  }, [openPane]);

  useEffect(() => {
    const closePane = (event) => {
      if (event.key === 'Escape') {
        setOpenPane(null);
      }

      if (event.target.closest(
        '.theme-doc-sidebar-container a, .theme-doc-toc-desktop a',
      )) {
        setOpenPane(null);
      }
    };

    document.addEventListener('click', closePane);
    document.addEventListener('keydown', closePane);

    return () => {
      document.removeEventListener('click', closePane);
      document.removeEventListener('keydown', closePane);
    };
  }, []);

  const togglePane = (pane) => {
    setOpenPane((current) => current === pane ? null : pane);
  };

  return (
    <>
      <button
        type="button"
        className="tablet-pane-toggle tablet-pane-toggle--left"
        aria-label="Toggle entries navigation"
        aria-expanded={openPane === 'left'}
        onClick={() => togglePane('left')}>
        <span aria-hidden="true">☰</span>
        <span>Entries</span>
      </button>
      <button
        type="button"
        className="tablet-pane-toggle tablet-pane-toggle--right"
        aria-label="Toggle table of contents"
        aria-expanded={openPane === 'right'}
        onClick={() => togglePane('right')}>
        <span>On this page</span>
        <span aria-hidden="true">☷</span>
      </button>
      {children}
    </>
  );
}

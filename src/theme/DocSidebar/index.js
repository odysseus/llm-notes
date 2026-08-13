import React from 'react';
import {useWindowSize} from '@docusaurus/theme-common';
import DocSidebarDesktop from '@theme/DocSidebar/Desktop';
import DocSidebarMobile from '@theme/DocSidebar/Mobile';

export default function DocSidebar(props) {
  const windowSize = useWindowSize();

  return (
    <>
      {/* Keep the entries available to the portrait tablet overlay. */}
      <DocSidebarDesktop {...props} />
      {/* Preserve Docusaurus's standard narrow-screen navbar navigation. */}
      {windowSize === 'mobile' && <DocSidebarMobile {...props} />}
    </>
  );
}

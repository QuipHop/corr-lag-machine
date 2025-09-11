import React, { useEffect, useRef } from 'react';
import { Network } from 'vis-network/standalone';
import type { CorrLagEdge } from '../types';


export default function GraphPanel({ edges }: { edges: CorrLagEdge[] }) {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!ref.current) return;
    // collect node ids
    const nodeIds = Array.from(new Set(edges.flatMap(e => [e.source, e.target])));
    const nodes = nodeIds.map(id => ({ id, label: id }));
    const dataEdges = edges.map(e => ({
      from: e.source, to: e.target, label: `lag ${e.lag}
${e.weight.toFixed(2)}`, width: 1 + Math.min(6, Math.abs(e.weight) * 6), color: { color: e.weight >= 0 ? '#3ddc84' : '#ff6b6b' }
    }));
    const network = new Network(ref.current, { nodes, edges: dataEdges }, { physics: { stabilization: true }, interaction: { hover: true } });
    return () => network.destroy();
  }, [edges]);
  return <div ref={ref} style={{ height: 420 }} className="card" />;
}
import React, { createContext, useContext, useMemo, useState } from 'react';
import type { SeriesIn } from '../api/analysis';

type Ctx = {
    selected: SeriesIn[];
    upsertSeries: (s: SeriesIn) => void; // додати або замінити за code
    removeByCode: (code: string) => void;
    clear: () => void;
};

const AnalysisSelectionContext = createContext<Ctx | null>(null);

export function AnalysisSelectionProvider({ children }: { children: React.ReactNode }) {
    const [selected, setSelected] = useState<SeriesIn[]>([]);

    const api = useMemo<Ctx>(() => ({
        selected,
        upsertSeries: (s) => {
            setSelected(prev => {
                const i = prev.findIndex(x => x.code === s.code);
                if (i === -1) return [...prev, s];
                const copy = prev.slice();
                copy[i] = s;
                return copy;
            });
        },
        removeByCode: (code) => setSelected(prev => prev.filter(x => x.code !== code)),
        clear: () => setSelected([]),
    }), [selected]);

    return (
        <AnalysisSelectionContext.Provider value={api}>
            {children}
        </AnalysisSelectionContext.Provider>
    );
}

export function useAnalysisSelection() {
    const ctx = useContext(AnalysisSelectionContext);
    if (!ctx) throw new Error('useAnalysisSelection must be used within AnalysisSelectionProvider');
    return ctx;
}

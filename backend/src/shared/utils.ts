/** Coerce body/query "seriesIds" of various shapes into string[] */
export const toStringIds = (ids: unknown): string[] => {
    if (Array.isArray(ids)) return ids.map(String);
    if (typeof ids === 'string') {
        // "a,b,c" or '["a","b"]' or '1,2,3'
        try {
            const parsed = JSON.parse(ids);
            if (Array.isArray(parsed)) return parsed.map(String);
        } catch { /* fall through */ }
        return ids.split(',').map(s => s.trim()).filter(Boolean);
    }
    return [];
};

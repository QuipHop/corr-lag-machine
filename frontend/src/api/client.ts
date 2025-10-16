const BASE = import.meta.env.VITE_API_BASE || 'http://localhost:3000';

type Query = Record<string, string | number | boolean>;

type ApiOptions<TBody = unknown> =
    Omit<RequestInit, "body"> & {
        query?: Query;
        body?: TBody; // ← дозволяємо звичайні об’єкти; далі самі JSON.stringify
    };

export async function api<TResp, TBody = unknown>(
    path: string,
    opts: ApiOptions<TBody> = {},
): Promise<TResp> {
    const url = new URL(path, BASE);
    if (opts.query) {
        for (const [k, v] of Object.entries(opts.query)) {
            url.searchParams.set(k, String(v));
        }
    }

    const {
        body,
        headers,
        method = "GET",
        ...rest
    } = opts;

    const payload =
        body === undefined
            ? undefined
            : (typeof body === "string" || body instanceof FormData ? body : JSON.stringify(body));

    const res = await fetch(url.toString(), {
        method,
        headers: {
            "Content-Type": payload instanceof FormData ? undefined as any : "application/json",
            ...(headers ?? {}),
        } as HeadersInit,
        body: payload,
        ...rest,
    });

    if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`${res.status} ${res.statusText} ${text}`);
    }
    return (await res.json()) as TResp;
}

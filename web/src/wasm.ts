export type WasmEngine = {
  new (): WasmEngine
  best_move(fen: string, depth: number): string
}

export async function initWasm(): Promise<{ WasmEngine: WasmEngine }> {
  // @ts-ignore
  const mod = await import('/pkg/tinyccrl_engine.js')
  await mod.default()
  return mod
}

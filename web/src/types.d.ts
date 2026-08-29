declare module '/pkg/tinyccrl_engine.js' {
  export default function init(): Promise<void>
  export class WasmEngine {
    constructor()
    best_move(fen: string, depth: number): string
  }
}

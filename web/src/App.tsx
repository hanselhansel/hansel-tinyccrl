import { useEffect, useState } from 'react'
import { Chess, type Square } from 'chess.js'
import { initWasm, type WasmEngine } from './wasm'

const PIECE_SYMBOLS: Record<string, string> = {
  p: '♟', n: '♞', b: '♝', r: '♜', q: '♛', k: '♚',
  P: '♙', N: '♘', B: '♗', R: '♖', Q: '♕', K: '♔',
}

function Board({ game, selected, onSquareClick }: {
  game: Chess
  selected: string | null
  onSquareClick: (sq: string) => void
}) {
  const files = 'abcdefgh'
  const ranks = '87654321'
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 48px)', border: '2px solid #333' }}>
      {ranks.split('').map((rank) =>
        files.split('').map((file) => {
          const sq = file + rank
          const piece = game.get(sq as Square)
          const isLight = (files.indexOf(file) + ranks.indexOf(rank)) % 2 === 0
          const isSelected = selected === sq
          return (
            <div
              key={sq}
              onClick={() => onSquareClick(sq)}
              style={{
                width: 48,
                height: 48,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 32,
                cursor: 'pointer',
                backgroundColor: isSelected ? '#aef' : isLight ? '#f0d9b5' : '#b58863',
              }}
            >
              {piece && PIECE_SYMBOLS[piece.color === 'w' ? piece.type.toUpperCase() : piece.type]}
            </div>
          )
        })
      )}
    </div>
  )
}

export default function App() {
  const [engine, setEngine] = useState<WasmEngine | null>(null)
  const [game, setGame] = useState(new Chess())
  const [selected, setSelected] = useState<string | null>(null)
  const [status, setStatus] = useState('Loading engine...')
  const [depth, setDepth] = useState(4)

  useEffect(() => {
    initWasm().then((mod) => {
      setEngine(new mod.WasmEngine())
      setStatus('Ready')
    })
  }, [])

  const onSquareClick = (sq: string) => {
    if (!engine || game.isGameOver()) return

    if (selected) {
      const moves = game.moves({ verbose: true })
      const move = moves.find((m) => m.from === selected && m.to === sq)
      if (move) {
        const g = new Chess(game.fen())
        g.move(move)
        setGame(g)
        setSelected(null)
        if (!g.isGameOver()) {
          setStatus('Thinking...')
          setTimeout(() => {
            const best = engine.best_move(g.fen(), depth)
            const reply = new Chess(g.fen())
            reply.move(best as Square)
            setGame(reply)
            setStatus(`Played ${best}`)
          }, 10)
        }
        return
      }
      setSelected(null)
    }

    const piece = game.get(sq as Square)
    if (piece && piece.color === game.turn()) {
      setSelected(sq)
    }
  }

  return (
    <div style={{ padding: 20, fontFamily: 'sans-serif' }}>
      <h1>TinyCCRL</h1>
      <p>{status}</p>
      <div style={{ marginBottom: 10 }}>
        <label>Depth: {depth} </label>
        <input type="range" min={1} max={8} value={depth} onChange={(e) => setDepth(Number(e.target.value))} />
      </div>
      <Board game={game} selected={selected} onSquareClick={onSquareClick} />
      <div style={{ marginTop: 10 }}>
        <button onClick={() => { setGame(new Chess()); setStatus('Ready') }}>Reset</button>
      </div>
      <pre style={{ marginTop: 20, fontSize: 12 }}>{game.ascii()}</pre>
    </div>
  )
}

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// El proxy evita CORS en desarrollo y, mas importante, hace que el frontend hable siempre
// con rutas relativas. En produccion la API y el frontend salen por el mismo tunel, asi
// que no hay una URL de API que configurar en ningun lado.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/auth': 'http://localhost:8000',
      '/videos': 'http://localhost:8000',
      '/jobs': 'http://localhost:8000',
      '/fightcards': 'http://localhost:8000',
      '/salud': 'http://localhost:8000',
    },
  },
  test: { environment: 'node' },
})

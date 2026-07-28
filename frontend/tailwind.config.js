/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Terminal palette taken from the Figma.
        base:    { DEFAULT: '#1E1E1E', deep: '#171717', panel: '#232323', raised: '#2A2A2A' },
        edge:    { DEFAULT: '#333333', light: '#3D3D3D' },
        mint:    { DEFAULT: '#7EE3B0', dim: '#4FBF89', glow: '#A8F0CC' },
        ink:     { DEFAULT: '#E8E8E8', muted: '#9A9A9A', faint: '#6B6B6B' },
        danger:  { DEFAULT: '#F0736A', dim: '#8C3A35' },
        warn:    { DEFAULT: '#E8B84B' },
        info:    { DEFAULT: '#8FA8E8' },
        accent:  { DEFAULT: '#9BA9F0' },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'SF Mono', 'Menlo', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      fontSize: {
        '2xs': ['0.625rem', { lineHeight: '0.875rem' }],
      },
    },
  },
  plugins: [],
}

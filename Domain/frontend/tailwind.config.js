/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'llm-dark': '#0f0f1a',
        'llm-accent': '#3B82F6',
      },
    },
  },
  plugins: [],
}
<script setup lang="ts">
withDefaults(defineProps<{ name: string; size?: number }>(), { size: 20 });
const paths: Record<string, string> = {
  overview: 'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z',
  data: 'M20 6c0 2.2-16 2.2-16 0s16-2.2 16 0v12c0 3-16 3-16 0V6 M4 12c0 3 16 3 16 0',
  factor: 'M3 17l5-5 4 3 8-10 M14 5h6v6 M3 21h18',
  feedback: 'M4 8a8 8 0 0 1 14-2l2 2 M20 3v5h-5 M20 16A8 8 0 0 1 6 18l-2-2 M4 21v-5h5',
  archive: 'M4 4h16v5H4z M5 9v11h14V9 M9 13h6',
  paper: 'M14 3H5v18h14V8z M14 3v5h5 M8 12h8 M8 16h6',
  plus: 'M12 5v14 M5 12h14',
  arrow: 'M5 12h14 M13 6l6 6-6 6',
  chevron: 'M9 5l7 7-7 7',
  expand: 'M8 3H3v5 M16 3h5v5 M21 16v5h-5 M8 21H3v-5',
  refresh: 'M20 7v5h-5 M4 17v-5h5 M6 6a8 8 0 0 1 14 6 M18 18a8 8 0 0 1-14-6',
  download: 'M12 3v12 M7 10l5 5 5-5 M4 16v5h16v-5',
  settings: 'M4 7h16 M4 17h16 M8 4v6 M16 14v6',
  check: 'M5 12l4 4L19 6',
  lock: 'M6 10h12v11H6z M8 10V6a4 4 0 0 1 8 0v4 M12 14v3',
  clock: 'M12 8v5l3 2 M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0',
  terminal: 'M4 5h16v14H4z M7 9l3 3-3 3 M13 15h4',
  search: 'M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16 M16 16l5 5',
  close: 'M6 6l12 12 M18 6L6 18',
  info: 'M12 11v6 M12 7h.01 M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0',
  model: 'M9 3h6l2 4 4 2v6l-4 2-2 4H9l-2-4-4-2V9l4-2z M9 9h6v6H9z'
};
</script>

<template>
  <svg :width="size" :height="size" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path :d="paths[name] || paths.factor" />
  </svg>
</template>

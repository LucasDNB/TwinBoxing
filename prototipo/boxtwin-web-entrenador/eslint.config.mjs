import next from "eslint-config-next";

// Lo que copio el CLI de shadcn (Animate UI y la base de shadcn, con sus hooks) es codigo de
// terceros: no se edita salvo por tokens (02 §3) y no pasa las reglas del React Compiler que
// trae eslint-config-next 16. Se excluye del lint en vez de reescribirlo; la prueba de humo
// verifica que monta (components/animate-ui/NOTAS.md).
const config = [
  ...next,
  {
    ignores: [
      "out/**", ".next/**", "playwright-report/**", "test-results/**", "next-env.d.ts",
      "components/animate-ui/**", "components/ui/**", "hooks/**",
    ],
  },
];

export default config;

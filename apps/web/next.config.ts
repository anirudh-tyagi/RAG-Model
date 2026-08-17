import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  // Emits a self-contained server bundle so the Docker image doesn't need
  // node_modules copied into it.
  output: "standalone",
  eslint: {
    // Linting is Biome's job here; `next build` shouldn't also try.
    ignoreDuringBuilds: true,
  },
};

export default config;

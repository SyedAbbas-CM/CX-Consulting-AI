"use client";

import { useEffect, useState } from "react";
import { getLlmConfig, getLlmRuntimeStatus, setLlmBackend, setAzureOpenAIConfig, AzureOpenAIConfigRequest, LlmBackendUpdateRequest } from "@/lib/apiClient";

export default function LlmSettingsPage() {
  const [cfg, setCfg] = useState<any | null>(null);
  const [runtime, setRuntime] = useState<any | null>(null);
  const [backendChoice, setBackendChoice] = useState<string>("llama.cpp");
  const [azureCfg, setAzureCfg] = useState<AzureOpenAIConfigRequest>({ endpoint: "", api_key: "", deployment: "" });
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const c = await getLlmConfig();
        setCfg(c);
        setBackendChoice(c.backend);
        const r = await getLlmRuntimeStatus();
        setRuntime(r);
      } catch (e: any) {
        console.error(e);
      }
    })();
  }, []);

  const saveChanges = async () => {
    setSaving(true);
    try {
      if (backendChoice === "azure") {
        await setAzureOpenAIConfig(azureCfg);
      }
      const payload: LlmBackendUpdateRequest = { backend: backendChoice };
      await setLlmBackend(payload);
      setMsg("Saved & reloaded");
    } catch (e: any) {
      setMsg(e.message || "Error");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-semibold mb-4">LLM Settings</h1>

      {msg && <p className="mb-4 text-sm text-green-600">{msg}</p>}

      <label className="block mb-2 font-medium">Backend</label>
      <select
        className="border rounded p-2 mb-4 w-full"
        value={backendChoice}
        onChange={(e) => setBackendChoice(e.target.value)}
      >
        <option value="llama.cpp">llama.cpp (local)</option>
        <option value="azure">Azure OpenAI</option>
        <option value="ollama">Ollama</option>
      </select>

      {backendChoice === "azure" && (
        <div className="space-y-3 mt-4">
          <input
            className="border rounded p-2 w-full"
            placeholder="Endpoint https://...azure.com"
            value={azureCfg.endpoint}
            onChange={(e) => setAzureCfg({ ...azureCfg, endpoint: e.target.value })}
          />
          <input
            className="border rounded p-2 w-full"
            placeholder="API Key"
            value={azureCfg.api_key}
            onChange={(e) => setAzureCfg({ ...azureCfg, api_key: e.target.value })}
          />
          <input
            className="border rounded p-2 w-full"
            placeholder="Deployment (e.g. gpt-4o-mini)"
            value={azureCfg.deployment}
            onChange={(e) => setAzureCfg({ ...azureCfg, deployment: e.target.value })}
          />
        </div>
      )}

      <button
        className="mt-6 bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
        onClick={saveChanges}
        disabled={saving}
      >
        {saving ? "Saving…" : "Save"}
      </button>

      {runtime && (
        <div className="mt-8 text-sm text-gray-700">
          <h2 className="font-medium">Runtime status</h2>
          <pre className="bg-gray-100 p-4 rounded overflow-x-auto text-xs mt-2">
            {JSON.stringify(runtime, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

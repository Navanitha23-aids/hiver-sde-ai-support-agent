const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:5001";

async function handle(res) {
  let body = null;
  try {
    body = await res.json();
  } catch {
    // no JSON body
  }
  if (!res.ok) {
    const message = (body && body.error) || `Request failed with status ${res.status}`;
    throw new Error(message);
  }
  return body;
}

export async function getHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  return handle(res);
}

export async function getBrands() {
  const res = await fetch(`${BASE_URL}/api/brands`);
  return handle(res);
}

export async function getIntents() {
  const res = await fetch(`${BASE_URL}/api/intents`);
  return handle(res);
}

export async function analyzeMessage(message) {
  const res = await fetch(`${BASE_URL}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  return handle(res);
}

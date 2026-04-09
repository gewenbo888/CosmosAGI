"""CosmosAGI — Web UI."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, jsonify, render_template_string, request

from cosmos_agi.config.settings import settings
from cosmos_agi.core.graph import run_agent_loop

# Configure
settings.safety.enable_human_in_the_loop = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

app = Flask(__name__)

# In-memory task store
tasks: dict[str, dict] = {}

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CosmosAGI</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    min-height: 100vh;
  }
  .container { max-width: 900px; margin: 0 auto; padding: 20px; }

  header {
    text-align: center;
    padding: 40px 0 30px;
    border-bottom: 1px solid #1a1a2e;
    margin-bottom: 30px;
  }
  header h1 {
    font-size: 2.2em;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }
  header p { color: #888; font-size: 0.95em; }

  .input-area {
    display: flex;
    gap: 10px;
    margin-bottom: 12px;
  }
  .input-area textarea {
    flex: 1;
    background: #12121a;
    border: 1px solid #2a2a3e;
    border-radius: 12px;
    color: #e0e0e0;
    padding: 14px 18px;
    font-size: 1em;
    font-family: inherit;
    resize: none;
    height: 56px;
    transition: border-color 0.2s;
  }
  .input-area textarea:focus {
    outline: none;
    border-color: #667eea;
  }
  .input-area button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0 28px;
    font-size: 1em;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.2s;
  }
  .input-area button:hover { opacity: 0.9; }
  .input-area button:disabled { opacity: 0.4; cursor: not-allowed; }

  .controls {
    display: flex;
    gap: 16px;
    margin-bottom: 30px;
    flex-wrap: wrap;
  }
  .controls label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.85em;
    color: #999;
    cursor: pointer;
  }
  .controls select, .controls input[type="number"] {
    background: #12121a;
    border: 1px solid #2a2a3e;
    color: #e0e0e0;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 0.85em;
  }
  .controls input[type="checkbox"] { accent-color: #667eea; }

  .task-card {
    background: #12121a;
    border: 1px solid #1a1a2e;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: border-color 0.3s;
  }
  .task-card.running { border-color: #667eea; }
  .task-card.complete { border-color: #4ade80; }
  .task-card.error { border-color: #f87171; }

  .task-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }
  .task-prompt {
    font-weight: 600;
    font-size: 1em;
    color: #fff;
  }
  .task-status {
    font-size: 0.8em;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
  }
  .status-running { background: #1e1b4b; color: #818cf8; }
  .status-complete { background: #052e16; color: #4ade80; }
  .status-error { background: #450a0a; color: #f87171; }

  .task-result {
    background: #0a0a12;
    border-radius: 8px;
    padding: 16px;
    margin-top: 12px;
    white-space: pre-wrap;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.88em;
    line-height: 1.6;
    color: #c0c0d0;
    max-height: 500px;
    overflow-y: auto;
  }
  .task-meta {
    display: flex;
    gap: 16px;
    margin-top: 10px;
    font-size: 0.8em;
    color: #666;
  }
  .task-meta span { display: flex; align-items: center; gap: 4px; }

  .spinner {
    display: inline-block;
    width: 14px; height: 14px;
    border: 2px solid #333;
    border-top-color: #818cf8;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-right: 6px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #444;
  }
  .empty-state p { font-size: 1.1em; margin-bottom: 8px; }
  .empty-state small { color: #333; }

  .examples {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 16px;
  }
  .examples button {
    background: #1a1a2e;
    border: 1px solid #2a2a3e;
    color: #aaa;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.82em;
    cursor: pointer;
    transition: all 0.2s;
  }
  .examples button:hover {
    border-color: #667eea;
    color: #fff;
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>CosmosAGI</h1>
    <p>Autonomous agent loop &bull; World model &bull; Multi-agent collaboration &bull; Self-learning</p>
  </header>

  <div class="input-area">
    <textarea id="taskInput" placeholder="Enter a task..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();submitTask()}"></textarea>
    <button id="submitBtn" onclick="submitTask()">Run</button>
  </div>

  <div class="controls">
    <label>
      Model:
      <select id="modelSelect">
        <option value="ollama/deepseek-r1:14b">DeepSeek R1 14B (local)</option>
        <option value="ollama/deepseek-r1:32b">DeepSeek R1 32B (local)</option>
        <option value="ollama/deepseek-r1:70b">DeepSeek R1 70B (local)</option>
        <option value="claude-sonnet-4-20250514">Claude Sonnet (API key)</option>
        <option value="gpt-4o">GPT-4o (API key)</option>
      </select>
    </label>
    <label>
      <input type="checkbox" id="teamMode"> Team Mode
    </label>
    <label>
      Max iterations:
      <input type="number" id="maxIter" value="3" min="1" max="20" style="width:50px">
    </label>
  </div>

  <div id="taskList">
    <div class="empty-state">
      <p>No tasks yet</p>
      <small>Enter a task above or try an example:</small>
      <div class="examples">
        <button onclick="fillExample('Explain quantum entanglement in simple terms')">Quantum entanglement</button>
        <button onclick="fillExample('What are the pros and cons of microservices vs monoliths?')">Microservices vs Monoliths</button>
        <button onclick="fillExample('Write a Python function to find all prime numbers up to N')">Prime numbers</button>
        <button onclick="fillExample('Plan a 3-day trip to Tokyo on a budget')">Tokyo trip</button>
      </div>
    </div>
  </div>
</div>

<script>
let taskCount = 0;
const pollers = {};

function fillExample(text) {
  document.getElementById('taskInput').value = text;
  document.getElementById('taskInput').focus();
}

async function submitTask() {
  const input = document.getElementById('taskInput');
  const btn = document.getElementById('submitBtn');
  const task = input.value.trim();
  if (!task) return;

  input.value = '';
  btn.disabled = true;
  btn.textContent = 'Running...';

  const body = {
    task: task,
    model: document.getElementById('modelSelect').value,
    team_mode: document.getElementById('teamMode').checked,
    max_iterations: parseInt(document.getElementById('maxIter').value) || 3,
  };

  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await res.json();
    addTaskCard(data.task_id, task);
    pollTask(data.task_id);
  } catch (e) {
    alert('Error: ' + e.message);
  }

  btn.disabled = false;
  btn.textContent = 'Run';
}

function addTaskCard(taskId, prompt) {
  const list = document.getElementById('taskList');
  // Remove empty state
  const empty = list.querySelector('.empty-state');
  if (empty) empty.remove();

  const card = document.createElement('div');
  card.className = 'task-card running';
  card.id = 'task-' + taskId;
  card.innerHTML = `
    <div class="task-header">
      <div class="task-prompt">${escapeHtml(prompt)}</div>
      <span class="task-status status-running"><span class="spinner"></span>Running</span>
    </div>
    <div class="task-result">Thinking...</div>
  `;
  list.prepend(card);
}

function pollTask(taskId) {
  pollers[taskId] = setInterval(async () => {
    try {
      const res = await fetch('/api/status/' + taskId);
      const data = await res.json();
      const card = document.getElementById('task-' + taskId);
      if (!card) return;

      if (data.status === 'complete') {
        clearInterval(pollers[taskId]);
        card.className = 'task-card complete';
        card.querySelector('.task-status').className = 'task-status status-complete';
        card.querySelector('.task-status').innerHTML = 'Complete';
        card.querySelector('.task-result').textContent = data.result || 'No result';

        let meta = `<div class="task-meta">`;
        meta += `<span>Iterations: ${data.iterations || '?'}</span>`;
        meta += `<span>Time: ${data.elapsed || '?'}s</span>`;
        meta += `</div>`;
        card.insertAdjacentHTML('beforeend', meta);
      } else if (data.status === 'error') {
        clearInterval(pollers[taskId]);
        card.className = 'task-card error';
        card.querySelector('.task-status').className = 'task-status status-error';
        card.querySelector('.task-status').innerHTML = 'Error';
        card.querySelector('.task-result').textContent = data.error || 'Unknown error';
      }
    } catch (e) {}
  }, 2000);
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.json
    task_text = data.get("task", "")
    model = data.get("model", "ollama/deepseek-r1:14b")
    team_mode = data.get("team_mode", False)
    max_iterations = data.get("max_iterations", 3)

    if not task_text:
        return jsonify({"error": "No task provided"}), 400

    task_id = f"task_{int(time.time())}_{len(tasks)}"
    tasks[task_id] = {
        "status": "running",
        "task": task_text,
        "model": model,
        "result": None,
        "error": None,
        "iterations": 0,
        "started": time.time(),
        "elapsed": None,
    }

    def run_in_background():
        try:
            settings.llm.model = model
            state = run_agent_loop(
                task_text,
                max_iterations=max_iterations,
                team_mode=team_mode,
                learn=True,
            )
            result = state.final_answer or "No answer produced."
            if isinstance(result, dict):
                result = json.dumps(result, indent=2)

            tasks[task_id]["status"] = "complete"
            tasks[task_id]["result"] = str(result)
            tasks[task_id]["iterations"] = state.iteration
            tasks[task_id]["elapsed"] = round(time.time() - tasks[task_id]["started"], 1)
        except Exception as e:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = str(e)
            tasks[task_id]["elapsed"] = round(time.time() - tasks[task_id]["started"], 1)

    thread = threading.Thread(target=run_in_background, daemon=True)
    thread.start()

    return jsonify({"task_id": task_id, "status": "running"})


@app.route("/api/status/<task_id>")
def api_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task)


@app.route("/api/stats")
def api_stats():
    from cosmos_agi.evaluation.learning_loop import LearningLoop
    loop = LearningLoop()
    return jsonify(loop.get_performance_summary())


if __name__ == "__main__":
    print("\n  CosmosAGI Web UI")
    print("  Open http://localhost:7860 in your browser\n")
    app.run(host="0.0.0.0", port=7860, debug=False)

"use strict";

const grid = document.getElementById("grid");
const notice = document.getElementById("notice");
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");
const cards = new Map();
const browserFeeds = new Map();
let lightboxSessionId = null;

function say(text) {
  notice.textContent = text || "";
}

function openLightbox(sessionId) {
  lightboxSessionId = sessionId;
  lightboxImg.src = `/stream/${sessionId}.mjpeg`;
  lightbox.hidden = false;
  document.body.classList.add("lightbox-open");
}

function closeLightbox() {
  if (lightbox.hidden) return;
  lightbox.hidden = true;
  lightboxImg.removeAttribute("src"); // 断开放大镜的 MJPEG 连接
  document.body.classList.remove("lightbox-open");
  lightboxSessionId = null;
}

// 浮层在预览图的 mouseup 中打开：若用 click 关闭，同一次点击会立刻把浮层关掉
// （浏览器在派发 click 时按当前位置命中测试）。改用 pointerdown，它必定早于
// 打开浮层的 mouseup，因此不会自触发。
lightbox.addEventListener("pointerdown", closeLightbox);
// 放大画面里也能直接拖拽框选：拦住图片上的 pointerdown，避免触发浮层的“点击关闭”。
lightboxImg.addEventListener("pointerdown", (event) => event.stopPropagation());
bindDragToAddBox(lightboxImg, () => lightboxSessionId, () => closeLightbox());
document.getElementById("lightbox-close").addEventListener("click", closeLightbox);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeLightbox();
});

async function api(path, options) {
  const response = await fetch(path, options);
  let data = {};
  try {
    data = await response.json();
  } catch (error) {
    data = {};
  }
  if (!response.ok) {
    throw new Error(data.error || response.statusText);
  }
  return data;
}

function parseSeeds(text) {
  const boxes = [];
  String(text)
    .split(/[;|]/)
    .forEach((group) => {
      const parts = group
        .split(",")
        .map((part) => part.trim())
        .filter((part) => part.length);
      if (parts.length !== 4) return;
      const values = parts.map(Number);
      if (values.every((value) => Number.isFinite(value))) boxes.push(values);
    });
  return boxes;
}

function dropFeed(sessionId) {
  const stop = browserFeeds.get(sessionId);
  if (stop) {
    stop();
    browserFeeds.delete(sessionId);
  }
}

function makeCard(session) {
  const card = document.createElement("section");
  card.className = "card";
  card.innerHTML = `
    <div class="card-head">
      <span class="name"></span>
      <span class="state"></span>
      <button class="stop" type="button">停止</button>
      <button class="restart" type="button">重启</button>
      <button class="remove" type="button">删除</button>
    </div>
    <img class="preview" alt="实时画面">
    <dl class="stats"></dl>
    <div class="paths"></div>
    <div class="hint">拖拽可框选新目标（下一帧生效）· 单击放大画面</div>
    <div class="recording" hidden>
      <input class="recording-path" placeholder="保存路径（留空使用默认）">
      <button class="save-recording" type="button">保存录像</button>
      <button class="discard-recording" type="button">删除录像</button>
    </div>
    <div class="error"></div>`;
  const img = card.querySelector(".preview");
  img.src = `/stream/${session.id}.mjpeg`;
  img.title = "单击放大，拖拽框选目标";

  card.querySelector(".stop").addEventListener("click", async () => {
    try {
      await api(`/api/sessions/${session.id}/stop`, { method: "POST" });
      say("已请求停止会话");
      refresh();
    } catch (error) {
      say(error.message);
    }
  });
  card.querySelector(".restart").addEventListener("click", async () => {
    if (card.dataset.recordingPending === "1" && !confirm("重启会丢弃本次未保存的录像，继续？")) {
      return;
    }
    try {
      await api(`/api/sessions/${session.id}/restart`, { method: "POST" });
      // 重启会替换服务端会话句柄并关闭旧 MJPEG 连接；流 URL 不变时浏览器
      // <img> 不会自动重新请求，画面会停在停止前的最后一帧，因此必须显式重连
      // （加时间戳参数避免缓存命中同一条死连接）。
      const streamUrl = `/stream/${session.id}.mjpeg?t=${Date.now()}`;
      img.src = streamUrl;
      if (!lightbox.hidden && lightboxSessionId === session.id) lightboxImg.src = streamUrl;
      say("已重启会话");
      refresh();
    } catch (error) {
      say(error.message);
    }
  });
  card.querySelector(".save-recording").addEventListener("click", async () => {
    const pathInput = card.querySelector(".recording-path");
    try {
      const data = await api(`/api/sessions/${session.id}/recording/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: pathInput.value.trim() || null }),
      });
      say(`录像已保存：${data.saved}`);
      refresh();
    } catch (error) {
      say(error.message);
    }
  });
  card.querySelector(".discard-recording").addEventListener("click", async () => {
    try {
      await api(`/api/sessions/${session.id}/recording/discard`, { method: "POST" });
      say("已删除预录视频");
      refresh();
    } catch (error) {
      say(error.message);
    }
  });
  card.querySelector(".remove").addEventListener("click", async () => {
    if (card.dataset.recordingPending === "1" && !confirm("该会话有未保存的录像，删除后将丢弃。继续？")) {
      return;
    }
    try {
      await api(`/api/sessions/${session.id}`, { method: "DELETE" });
      say("已删除会话");
    } catch (error) {
      say(error.message);
    }
    dropFeed(session.id);
    card.remove();
    cards.delete(session.id);
    refresh();
  });

  bindDragToAddBox(img, () => session.id, () => openLightbox(session.id));
  grid.appendChild(card);
  cards.set(session.id, card);
  return card;
}

function bindDragToAddBox(img, getSessionId, onTap) {
  let start = null;
  const toImageCoords = (event) => {
    const rect = img.getBoundingClientRect();
    const naturalWidth = img.naturalWidth || rect.width;
    const naturalHeight = img.naturalHeight || rect.height;
    return [
      ((event.clientX - rect.left) / rect.width) * naturalWidth,
      ((event.clientY - rect.top) / rect.height) * naturalHeight,
    ];
  };
  img.addEventListener("mousedown", (event) => {
    start = toImageCoords(event);
    event.preventDefault();
  });
  img.addEventListener("mouseup", async (event) => {
    if (!start) return;
    const [x1, y1] = start;
    const [x2, y2] = toImageCoords(event);
    start = null;
    const bbox = [Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1)];
    if (bbox[2] < 8 || bbox[3] < 8) {
      // 没有拖动 → 视为单击（预览图放大 / 放大镜退出）
      if (onTap) onTap();
      return;
    }
    const sessionId = getSessionId();
    if (!sessionId) return;
    try {
      await api(`/api/sessions/${sessionId}/box`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bbox }),
      });
      say("已添加目标框，下一帧生效");
    } catch (error) {
      say(error.message);
    }
  });
}

function updateCard(session) {
  const card = cards.get(session.id) || makeCard(session);
  const running = session.state === "running" || session.state === "starting";
  card.querySelector(".name").textContent = session.name || session.id;
  card.querySelector(".state").textContent = session.state;
  card.querySelector(".stats").innerHTML = `
    <dt>帧</dt><dd>${session.frames ?? 0}</dd>
    <dt>帧率</dt><dd>CAP ${(session.capture_fps ?? 0).toFixed(1)} / PROC ${(session.process_fps ?? 0).toFixed(1)} / STR ${(session.stream_fps ?? 0).toFixed(1)}</dd>
    <dt>在线 ID</dt><dd>${session.active ?? 0}</dd>
    <dt>新建 ID</dt><dd>${session.created_ids ?? 0}</dd>
    <dt>销毁 ID</dt><dd>${session.removed_ids ?? 0}</dd>
    <dt>轨迹点</dt><dd>${session.trail_points ?? 0}</dd>
    <dt>事件</dt><dd>${session.events ?? 0}</dd>
    <dt>预录</dt><dd>${session.recording_frames ?? 0} 帧</dd>`;
  card.querySelector(".paths").textContent =
    `CSV ${session.csv || "-"} · JSONL ${session.jsonl || "-"}`;
  card.querySelector(".error").textContent = session.error || "";
  const pendingRecording = Boolean(session.recording_pending);
  card.dataset.recordingPending = pendingRecording ? "1" : "";
  const recording = card.querySelector(".recording");
  recording.hidden = running || !pendingRecording;
  if (pendingRecording) {
    const pathInput = recording.querySelector(".recording-path");
    if (!pathInput.dataset.initialised) {
      pathInput.value = session.recording_target || "";
      pathInput.dataset.initialised = "1";
    }
  }
  card.querySelector(".stop").style.display = running ? "" : "none";
  card.querySelector(".restart").style.display = running ? "none" : "";
  card.querySelector(".remove").style.display = running ? "none" : "";
}

async function refresh() {
  try {
    const data = await api("/api/sessions");
    data.sessions.forEach(updateCard);
    const live = new Set(data.sessions.map((session) => session.id));
    cards.forEach((card, id) => {
      if (!live.has(id)) {
        if (lightboxSessionId === id) closeLightbox();
        dropFeed(id);
        card.remove();
        cards.delete(id);
      }
    });
    if (!data.sessions.length) {
      say("暂无会话：选择输入源后点“启动”，或用“上传视频 / 浏览器摄像头”。");
    }
  } catch (error) {
    say(`无法连接服务端：${error.message}`);
  }
}

document.getElementById("add-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const kind = document.getElementById("kind").value;
  const spec = document.getElementById("spec").value.trim();
  const seedsText = document.getElementById("seeds").value.trim();
  const seedBoxes = parseSeeds(seedsText);
  if (seedsText && !seedBoxes.length) {
    say("种子框格式应为 x,y,w,h（多组用 ; 分隔）");
    return;
  }
  try {
    await api("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind, spec: spec || null, seed_boxes: seedBoxes }),
    });
    say(seedBoxes.length ? `已启动会话（${seedBoxes.length} 个种子框）` : "已启动会话");
    refresh();
  } catch (error) {
    say(error.message);
  }
});

document.getElementById("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.getElementById("file");
  if (!input.files.length) {
    say("请先选择一个视频文件");
    return;
  }
  const body = new FormData();
  body.append("file", input.files[0]);
  try {
    await api("/api/upload", { method: "POST", body });
    say("上传完成，正在后台处理");
    refresh();
  } catch (error) {
    say(error.message);
  }
});

document.getElementById("browser-cam").addEventListener("click", async () => {
  try {
    const handle = await api("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: "browser" }),
    });
    await startBrowserFeed(handle.id);
    say("浏览器摄像头已开启（约 8 fps，需授权摄像头访问）");
    refresh();
  } catch (error) {
    say(`浏览器摄像头不可用：${error.message}`);
  }
});

async function startBrowserFeed(sessionId) {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  const video = document.createElement("video");
  video.srcObject = stream;
  video.muted = true;
  await video.play();
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  let running = true;
  browserFeeds.set(sessionId, () => {
    running = false;
    stream.getTracks().forEach((track) => track.stop());
  });
  const pump = async () => {
    if (!running) return;
    if (video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.7));
      if (blob && running) {
        try {
          await fetch(`/api/sessions/${sessionId}/frame`, {
            method: "POST",
            headers: { "Content-Type": "image/jpeg" },
            body: blob,
          });
        } catch (error) {
          /* 单帧失败忽略，下一帧继续 */
        }
      }
    }
    if (running) setTimeout(pump, 125);
  };
  pump();
}

window.addEventListener("beforeunload", () => {
  browserFeeds.forEach((stop) => stop());
});

refresh();
setInterval(refresh, 1000);

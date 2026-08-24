(function (root) {
  "use strict";

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function safeHref(value) {
    const href = value.trim();
    if (
      /^https?:\/\//i.test(href) ||
      /^mailto:/i.test(href) ||
      /^\/(?!\/)/.test(href) ||
      href.startsWith("#")
    ) {
      return escapeHtml(href);
    }
    return null;
  }

  function renderInline(source) {
    const tokens = [];
    const stash = (html) => {
      const token = `\uE000${tokens.length}\uE001`;
      tokens.push(html);
      return token;
    };

    let value = String(source);
    value = value.replace(/`([^`\n]+)`/g, (_, code) =>
      stash(`<code>${escapeHtml(code)}</code>`)
    );
    value = value.replace(/<((?:https?:\/\/|mailto:)[^>\s]+)>/gi, (_, href) => {
      const safe = safeHref(href);
      return safe
        ? stash(`<a href="${safe}" target="_blank" rel="noopener noreferrer">${escapeHtml(href)}</a>`)
        : escapeHtml(`<${href}>`);
    });
    value = value.replace(/\[([^\]\n]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (_, label, href) => {
      const safe = safeHref(href);
      return safe
        ? stash(`<a href="${safe}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a>`)
        : `${escapeHtml(label)} (${escapeHtml(href)})`;
    });

    value = escapeHtml(value);
    value = value.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    value = value.replace(/__([^_\n]+)__/g, "<strong>$1</strong>");
    value = value.replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
    value = value.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
    value = value.replace(/(^|[^_])_([^_\n]+)_(?!_)/g, "$1<em>$2</em>");
    value = value.replace(/  \n/g, "<br>");
    value = value.replace(/\uE000(\d+)\uE001/g, (_, index) => tokens[Number(index)]);
    return value;
  }

  function splitTableRow(line) {
    let value = line.trim();
    if (value.startsWith("|")) value = value.slice(1);
    if (value.endsWith("|")) value = value.slice(0, -1);
    const cells = [];
    let cell = "";
    let escaped = false;
    for (const character of value) {
      if (escaped) {
        cell += character;
        escaped = false;
      } else if (character === "\\") {
        escaped = true;
      } else if (character === "|") {
        cells.push(cell.trim());
        cell = "";
      } else {
        cell += character;
      }
    }
    cells.push(cell.trim());
    return cells;
  }

  function tableAlignments(line) {
    const cells = splitTableRow(line);
    if (!cells.length || cells.some((cell) => !/^:?-{3,}:?$/.test(cell))) return null;
    return cells.map((cell) => {
      if (cell.startsWith(":") && cell.endsWith(":")) return "center";
      if (cell.endsWith(":")) return "right";
      if (cell.startsWith(":")) return "left";
      return "";
    });
  }

  function alignmentAttribute(alignment) {
    return alignment ? ` style="text-align:${alignment}"` : "";
  }

  function isBlockStart(lines, index) {
    const line = lines[index] || "";
    if (!line.trim()) return true;
    if (/^\s*```/.test(line)) return true;
    if (/^\s{0,3}#{1,6}\s+/.test(line)) return true;
    if (/^\s{0,3}>\s?/.test(line)) return true;
    if (/^\s{0,3}(?:[-+*]|\d+\.)\s+/.test(line)) return true;
    if (/^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) return true;
    return index + 1 < lines.length && tableAlignments(lines[index + 1]) !== null;
  }

  function renderMarkdown(source) {
    const lines = String(source).replace(/\r\n?/g, "\n").split("\n");
    const output = [];
    let index = 0;

    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) {
        index += 1;
        continue;
      }

      const fence = line.match(/^\s*```\s*([\w+-]*)\s*$/);
      if (fence) {
        const code = [];
        index += 1;
        while (index < lines.length && !/^\s*```\s*$/.test(lines[index])) {
          code.push(lines[index]);
          index += 1;
        }
        if (index < lines.length) index += 1;
        const language = fence[1]
          ? ` class="language-${escapeHtml(fence[1])}"`
          : "";
        output.push(`<pre><code${language}>${escapeHtml(code.join("\n"))}</code></pre>`);
        continue;
      }

      const heading = line.match(/^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
      if (heading) {
        const level = heading[1].length;
        output.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
        index += 1;
        continue;
      }

      if (/^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
        output.push("<hr>");
        index += 1;
        continue;
      }

      if (/^\s{0,3}>\s?/.test(line)) {
        const quoted = [];
        while (index < lines.length && /^\s{0,3}>\s?/.test(lines[index])) {
          quoted.push(lines[index].replace(/^\s{0,3}>\s?/, ""));
          index += 1;
        }
        output.push(`<blockquote>${renderMarkdown(quoted.join("\n"))}</blockquote>`);
        continue;
      }

      const unordered = line.match(/^\s{0,3}[-+*]\s+(.+)$/);
      const ordered = line.match(/^\s{0,3}(\d+)\.\s+(.+)$/);
      if (unordered || ordered) {
        const tag = unordered ? "ul" : "ol";
        const start = ordered && Number(ordered[1]) !== 1 ? ` start="${Number(ordered[1])}"` : "";
        const items = [];
        const matcher = unordered
          ? /^\s{0,3}[-+*]\s+(.+)$/
          : /^\s{0,3}(\d+)\.\s+(.+)$/;
        while (index < lines.length) {
          const match = lines[index].match(matcher);
          if (!match) break;
          const item = match[match.length - 1];
          const task = item.match(/^\[([ xX])\]\s+(.+)$/);
          if (task) {
            const checked = task[1].toLowerCase() === "x" ? " checked" : "";
            items.push(`<li class="task-item"><input type="checkbox" disabled${checked}> ${renderInline(task[2])}</li>`);
          } else {
            items.push(`<li>${renderInline(item)}</li>`);
          }
          index += 1;
        }
        output.push(`<${tag}${start}>${items.join("")}</${tag}>`);
        continue;
      }

      const alignments = index + 1 < lines.length ? tableAlignments(lines[index + 1]) : null;
      if (alignments) {
        const headers = splitTableRow(line);
        index += 2;
        const rows = [];
        while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
          rows.push(splitTableRow(lines[index]));
          index += 1;
        }
        const headerHtml = headers
          .map((cell, cellIndex) => `<th${alignmentAttribute(alignments[cellIndex] || "")}>${renderInline(cell)}</th>`)
          .join("");
        const bodyHtml = rows
          .map((row) => `<tr>${headers.map((_, cellIndex) => `<td${alignmentAttribute(alignments[cellIndex] || "")}>${renderInline(row[cellIndex] || "")}</td>`).join("")}</tr>`)
          .join("");
        output.push(`<div class="table-wrap"><table><thead><tr>${headerHtml}</tr></thead><tbody>${bodyHtml}</tbody></table></div>`);
        continue;
      }

      const paragraph = [line];
      index += 1;
      while (index < lines.length && !isBlockStart(lines, index)) {
        paragraph.push(lines[index]);
        index += 1;
      }
      const joined = paragraph
        .map((part, partIndex) => {
          if (partIndex === paragraph.length - 1) return part;
          return part.endsWith("  ") ? `${part}\n` : `${part} `;
        })
        .join("");
      output.push(`<p>${renderInline(joined)}</p>`);
    }

    return output.join("\n");
  }

  function renderMarkdownInto(element, source) {
    element.dataset.rawMarkdown = String(source);
    element.innerHTML = renderMarkdown(source);
  }

  root.renderMarkdown = renderMarkdown;
  root.renderMarkdownInto = renderMarkdownInto;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { renderMarkdown };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);

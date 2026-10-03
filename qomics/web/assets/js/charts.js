/* =============================================================================
   Q-OMICS — charts.js
   Hand-written SVG. No libraries, no network. Every chart is responsive
   (re-rendered on container resize) and degrades to an explicit empty state.
   All DOM is built with createElement/textContent; no innerHTML.
   ========================================================================== */
(function (global) {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';

  /* Restrained categorical set. Desaturated; no neon. */
  var SERIES = [
    '#4FD1C5', '#5AA9C7', '#D98B5F', '#6FA88A',
    '#8A929C', '#C4A661', '#7E8B9B', '#A87F8E'
  ];

  var INK = '#E6E9ED';
  var MUTED = '#8A929C';
  var FAINT = '#656D77';
  var GRID = '#1B1F25';
  var AXIS = '#2E343D';
  var MONO = 'ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace';
  var UI = 'Inter, -apple-system, "Segoe UI", system-ui, sans-serif';

  /* ---------------------------------------------------------------- helpers */

  function isNum(v) {
    return typeof v === 'number' && isFinite(v);
  }

  function numOrNull(v) {
    if (v === null || v === undefined || v === '') return null;
    var n = typeof v === 'number' ? v : parseFloat(v);
    return isNum(n) ? n : null;
  }

  function fmt(v, dp) {
    var n = numOrNull(v);
    if (n === null) return '—';
    if (dp === undefined) dp = 3;
    return n.toFixed(dp);
  }

  function fmtInt(v) {
    var n = numOrNull(v);
    if (n === null) return '—';
    return Math.round(n).toLocaleString('en-US');
  }

  function h(tag, attrs, children) {
    var node = document.createElement(tag);
    if (attrs) {
      for (var k in attrs) {
        if (!Object.prototype.hasOwnProperty.call(attrs, k)) continue;
        var v = attrs[k];
        if (v === null || v === undefined || v === false) continue;
        if (k === 'class') node.className = v;
        else if (k === 'text') node.textContent = String(v);
        else if (k === 'html') throw new Error('innerHTML is not permitted');
        else if (k.slice(0, 2) === 'on' && typeof v === 'function') node.addEventListener(k.slice(2), v);
        else if (v === true) node.setAttribute(k, '');
        else node.setAttribute(k, String(v));
      }
    }
    append(node, children);
    return node;
  }

  function s(tag, attrs, children) {
    var node = document.createElementNS(NS, tag);
    if (attrs) {
      for (var k in attrs) {
        if (!Object.prototype.hasOwnProperty.call(attrs, k)) continue;
        var v = attrs[k];
        if (v === null || v === undefined || v === false) continue;
        if (k === 'text') { node.textContent = String(v); continue; }
        if (k === 'class') { node.setAttribute('class', v); continue; }
        node.setAttribute(k, String(v));
      }
    }
    append(node, children);
    return node;
  }

  function append(node, children) {
    if (children === null || children === undefined) return;
    if (!Array.isArray(children)) children = [children];
    for (var i = 0; i < children.length; i++) {
      var c = children[i];
      if (c === null || c === undefined || c === false) continue;
      node.appendChild(typeof c === 'object' && c.nodeType ? c : document.createTextNode(String(c)));
    }
  }

  function svgRoot(w, hgt) {
    return s('svg', {
      viewBox: '0 0 ' + w + ' ' + hgt,
      width: '100%',
      role: 'img',
      focusable: 'false',
      preserveAspectRatio: 'xMinYMin meet'
    });
  }

  function measure(host) {
    var w = host.clientWidth || host.getBoundingClientRect().width || 0;
    if (!isNum(w) || w < 240) w = 640;
    return Math.max(300, Math.min(1000, Math.round(w)));
  }

  function emptyState(host, message) {
    host.replaceChildren(h('div', { class: 'chart-empty', role: 'status' }, message || 'No data available.'));
  }

  /* Make a chart re-render when its host width changes. */
  function responsive(host, draw) {
    var state = { w: 0, ro: null };
    function run() {
      var w = measure(host);
      if (w === state.w && host.firstChild) return;
      state.w = w;
      draw(w);
    }
    run();
    if (typeof ResizeObserver === 'function') {
      state.ro = new ResizeObserver(function () { run(); });
      state.ro.observe(host);
    } else if (typeof global.addEventListener === 'function') {
      var raf = null;
      global.addEventListener('resize', function () {
        if (raf) return;
        raf = (global.requestAnimationFrame || setTimeout)(function () { raf = null; state.w = 0; run(); }, 120);
      });
    }
    return {
      redraw: function () { state.w = 0; run(); },
      destroy: function () { if (state.ro) { state.ro.disconnect(); state.ro = null; } host.replaceChildren(); }
    };
  }

  function tooltip(host) {
    var tip = h('div', { class: 'chart-tip', role: 'presentation' });
    host.appendChild(tip);
    return {
      show: function (x, y, rows) {
        tip.replaceChildren();
        append(tip, rows);
        tip.style.left = x + 'px';
        tip.style.top = y + 'px';
        tip.setAttribute('data-show', '1');
      },
      hide: function () { tip.removeAttribute('data-show'); }
    };
  }

  function legend(items) {
    if (!items.length) return null;
    return h('div', { class: 'chart-legend' }, items.map(function (it) {
      return h('span', { class: 'legend-item' }, [
        h('span', {
          class: 'legend-swatch' + (it.dashed ? ' dashed' : ''),
          style: it.dashed ? '' : 'background:' + it.color
        }),
        h('span', { text: it.label })
      ]);
    }));
  }

  function axisTicks(lo, hi, count) {
    var out = [], i, v;
    for (i = 0; i <= count; i++) {
      v = lo + (hi - lo) * (i / count);
      out.push(Math.round(v * 1000) / 1000);
    }
    return out;
  }

  /* ===================================================================== ROC */

  function roc(host, data, opts) {
    opts = opts || {};
    var curves = (data && Array.isArray(data.curves)) ? data.curves : [];

    var usable = curves.filter(function (c) {
      return c && Array.isArray(c.fpr) && Array.isArray(c.tpr) &&
        c.fpr.length > 1 && c.fpr.length === c.tpr.length;
    });

    if (!usable.length) {
      emptyState(host, 'No ROC data. /api/metrics returned no cross-validated curves.');
      return { redraw: function () {}, destroy: function () {} };
    }

    var primary = opts.primaryName || '';

    return responsive(host, function (W) {
      var H = Math.max(320, Math.min(520, Math.round(W * 0.72)));
      var m = { t: 14, r: 18, b: 46, l: 54 };
      var pw = W - m.l - m.r;
      var ph = H - m.t - m.b;
      if (pw < 80 || ph < 80) { emptyState(host, 'Container too narrow to plot.'); return; }

      var X = function (v) { return m.l + v * pw; };
      var Y = function (v) { return m.t + (1 - v) * ph; };

      var svg = svgRoot(W, H);
      var gridG = s('g', { 'aria-hidden': 'true' });
      var axisG = s('g', { 'aria-hidden': 'true' });
      var dataG = s('g');
      var hitG = s('g');

      var i, t, v, px, py;

      /* gridlines + ticks */
      var ticks = axisTicks(0, 1, 4);
      for (i = 0; i < ticks.length; i++) {
        v = ticks[i];
        px = X(v); py = Y(v);
        gridG.appendChild(s('line', { x1: m.l, y1: py, x2: m.l + pw, y2: py, stroke: GRID, 'stroke-width': 1 }));
        gridG.appendChild(s('line', { x1: px, y1: m.t, x2: px, y2: m.t + ph, stroke: GRID, 'stroke-width': 1 }));

        axisG.appendChild(s('text', {
          x: px, y: m.t + ph + 20, fill: FAINT, 'font-family': MONO, 'font-size': 12,
          'text-anchor': 'middle', text: v.toFixed(2)
        }));
        axisG.appendChild(s('text', {
          x: m.l - 10, y: py + 4, fill: FAINT, 'font-family': MONO, 'font-size': 12,
          'text-anchor': 'end', text: v.toFixed(2)
        }));
      }

      /* frame */
      axisG.appendChild(s('line', { x1: m.l, y1: m.t + ph, x2: m.l + pw, y2: m.t + ph, stroke: AXIS, 'stroke-width': 1 }));
      axisG.appendChild(s('line', { x1: m.l, y1: m.t, x2: m.l, y2: m.t + ph, stroke: AXIS, 'stroke-width': 1 }));

      /* axis titles */
      axisG.appendChild(s('text', {
        x: m.l + pw / 2, y: H - 8, fill: MUTED, 'font-family': UI, 'font-size': 12,
        'text-anchor': 'middle', text: 'False positive rate'
      }));
      axisG.appendChild(s('text', {
        x: 14, y: m.t + ph / 2, fill: MUTED, 'font-family': UI, 'font-size': 12,
        'text-anchor': 'middle', transform: 'rotate(-90 14 ' + (m.t + ph / 2) + ')', text: 'True positive rate'
      }));

      /* chance reference */
      dataG.appendChild(s('line', {
        x1: X(0), y1: Y(0), x2: X(1), y2: Y(1),
        stroke: FAINT, 'stroke-width': 1, 'stroke-dasharray': '3 4', 'aria-hidden': 'true'
      }));

      /* series */
      var series = [];
      var stride = Math.max(1, Math.ceil(usable[0].fpr.length / 600));
      for (i = 0; i < usable.length; i++) {
        var c = usable[i];
        var color = SERIES[i % SERIES.length];
        var isPrimary = primary && c.name === primary;
        var pts = [];
        for (var j = 0; j < c.fpr.length; j += stride) {
          var fx = numOrNull(c.fpr[j]), ty = numOrNull(c.tpr[j]);
          if (fx === null || ty === null) continue;
          pts.push([fx, ty]);
        }
        if (!pts.length) continue;
        series.push({ name: String(c.name == null ? 'model ' + (i + 1) : c.name), color: color, pts: pts, primary: !!isPrimary });

        var d = pts.map(function (p, k) {
          return (k ? 'L' : 'M') + X(p[0]).toFixed(2) + ' ' + Y(p[1]).toFixed(2);
        }).join(' ');

        dataG.appendChild(s('path', {
          d: d, fill: 'none', stroke: color,
          'stroke-width': isPrimary ? 2.1 : 1.3,
          'stroke-linejoin': 'round', 'stroke-linecap': 'round',
          opacity: isPrimary ? 1 : 0.78
        }));
      }

      if (!series.length) { emptyState(host, 'ROC payload contained no finite points.'); return; }

      /* hover surface + markers */
      var hovLine = s('line', { x1: 0, y1: m.t, x2: 0, y2: m.t + ph, stroke: AXIS, 'stroke-width': 1, opacity: 0, 'aria-hidden': 'true' });
      hitG.appendChild(hovLine);
      var dots = series.map(function (se) {
        var d = s('circle', { r: 3, fill: se.color, stroke: '#0A0C0F', 'stroke-width': 1, opacity: 0, 'aria-hidden': 'true' });
        hitG.appendChild(d);
        return d;
      });

      var capture = s('rect', {
        x: m.l, y: m.t, width: pw, height: ph, fill: 'transparent', style: 'cursor:crosshair'
      });
      hitG.appendChild(capture);

      svg.appendChild(gridG);
      svg.appendChild(dataG);
      svg.appendChild(axisG);
      svg.appendChild(hitG);

      host.replaceChildren(svg);
      var tip = tooltip(host);

      function nearest(ev) {
        var rect = svg.getBoundingClientRect();
        if (!rect.width) return null;
        var scale = W / rect.width;
        var cx = (ev.clientX - rect.left) * scale;
        var fpr = Math.max(0, Math.min(1, (cx - m.l) / pw));
        var rows = [];
        var best = null;
        for (var k = 0; k < series.length; k++) {
          var se = series[k];
          var bi = 0, bd = Infinity;
          for (var q = 0; q < se.pts.length; q++) {
            var dd = Math.abs(se.pts[q][0] - fpr);
            if (dd < bd) { bd = dd; bi = q; }
          }
          var pt = se.pts[bi];
          var sy = Y(pt[1]);
          rows.push({ name: se.name, color: se.color, fpr: pt[0], tpr: pt[1], y: sy });
          dots[k].setAttribute('cx', X(pt[0]));
          dots[k].setAttribute('cy', sy);
          dots[k].setAttribute('opacity', 1);
          if (best === null || pt[1] > best.tpr) best = { x: X(pt[0]), y: sy, fpr: pt[0], tpr: pt[1] };
        }
        if (!best) return null;
        hovLine.setAttribute('x1', best.x);
        hovLine.setAttribute('x2', best.x);
        hovLine.setAttribute('opacity', 1);

        var inv = rect.width / W;
        var html = [h('div', { class: 'tip-row' }, [
          h('span', { class: 'tip-name', text: 'fpr' }), h('span', { text: best.fpr.toFixed(3) })
        ])];
        for (var r = 0; r < rows.length; r++) {
          html.push(h('div', { class: 'tip-row' }, [
            h('span', { class: 'legend-swatch', style: 'background:' + rows[r].color }),
            h('span', { class: 'tip-name', text: 'tpr ' + rows[r].tpr.toFixed(3) })
          ]));
        }
        tip.show(best.x * inv, best.y * inv, html);
      }

      function clearHover() {
        hovLine.setAttribute('opacity', 0);
        dots.forEach(function (d) { d.setAttribute('opacity', 0); });
        tip.hide();
      }

      capture.addEventListener('pointermove', nearest);
      capture.addEventListener('pointerleave', clearHover);

      var items = [{ label: 'Chance (AUC 0.50)', dashed: true, color: FAINT }].concat(
        series.map(function (se) { return { label: se.name, color: se.color }; })
      );
      var lg = legend(items);
      if (lg) host.appendChild(lg);

      svg.setAttribute('aria-label',
        'Receiver operating characteristic curves for ' + series.length + ' model' + (series.length === 1 ? '' : 's') + '.');
    });
  }

  /* ====================================================== horizontal bar set */

  function hbars(host, items, opts) {
    opts = opts || {};
    var rows = (Array.isArray(items) ? items : []).filter(function (it) {
      return it && numOrNull(it.value) !== null;
    });
    if (!rows.length) {
      emptyState(host, opts.emptyMessage || 'No data available.');
      return { redraw: function () {}, destroy: function () {} };
    }

    return responsive(host, function (W) {
      var rowH = 30;
      var barH = 14;
      var m = { t: 6, r: 8, b: 20, l: 0 };
      var labelW = Math.max(72, Math.min(200, Math.round(W * 0.36)));
      var valueW = 62;
      var trackX = labelW + 10;
      var trackW = Math.max(40, W - trackX - valueW - 4);
      var H = m.t + m.b + rows.length * rowH;

      var maxV = 0;
      rows.forEach(function (r) {
        var v = Math.abs(numOrNull(r.value));
        if (v > maxV) maxV = v;
      });
      if (!isNum(maxV) || maxV <= 0) maxV = 1;

      var svg = svgRoot(W, H);
      var axisY = m.t + rows.length * rowH;

      /* baseline */
      svg.appendChild(s('line', {
        x1: trackX, y1: m.t, x2: trackX, y2: axisY,
        stroke: AXIS, 'stroke-width': 1, 'aria-hidden': 'true'
      }));

      var tip = null;
      rows.forEach(function (r, i) {
        var v = numOrNull(r.value);
        var frac = Math.abs(v) / maxV;
        var w = Math.max(1, frac * trackW);
        var y = m.t + i * rowH + (rowH - barH) / 2;
        var color = r.color || opts.color || SERIES[0];

        var g = s('g', { style: 'cursor:default' });
        g.appendChild(s('text', {
          x: 0, y: y + barH - 2, fill: INK, 'font-family': UI, 'font-size': 13,
          text: String(r.label == null ? '' : r.label)
        }));
        g.appendChild(s('rect', {
          x: trackX, y: y, width: trackW, height: barH,
          fill: '#171B21', rx: 1
        }));
        g.appendChild(s('rect', {
          x: trackX, y: y, width: w, height: barH, fill: color, rx: 1
        }));
        g.appendChild(s('text', {
          x: trackX + trackW + 6, y: y + barH - 2,
          fill: MUTED, 'font-family': MONO, 'font-size': 12, 'text-anchor': 'start',
          text: r.valueText != null ? String(r.valueText) : fmt(v, 3)
        }));
        svg.appendChild(g);
      });

      if (opts.axisLabel) {
        svg.appendChild(s('text', {
          x: trackX, y: H - 4, fill: FAINT, 'font-family': UI, 'font-size': 12,
          text: opts.axisLabel
        }));
      }

      host.replaceChildren(svg);
      void tip;
    });
  }

  /* ================================================ grouped vertical columns */

  function grouped(host, data, opts) {
    opts = opts || {};
    var groups = (Array.isArray(data && data.series) ? data.series : []).filter(function (se) {
      return se && Array.isArray(se.values) && se.values.length > 0;
    });
    if (!groups.length) {
      emptyState(host, opts.emptyMessage || 'No amplitude data available.');
      return { redraw: function () {}, destroy: function () {} };
    }

    var cats = Array.isArray(data.labels) && data.labels.length
      ? data.labels
      : groups[0].values.map(function (_, i) { return 'a' + i; });

    return responsive(host, function (W) {
      var m = { t: 16, r: 12, b: 42, l: 46 };
      var H = Math.max(200, Math.min(300, Math.round(W * 0.52)));
      var pw = W - m.l - m.r;
      var ph = H - m.t - m.b;
      if (pw < 60 || ph < 60) { emptyState(host, 'Container too narrow to plot.'); return; }

      var lo = 0, hi = 0;
      groups.forEach(function (se) {
        se.values.forEach(function (v) {
          var n = numOrNull(v);
          if (n === null) return;
          if (n < lo) lo = n;
          if (n > hi) hi = n;
        });
      });
      if (hi === lo) hi = lo + 1;
      var pad = (hi - lo) * 0.12;
      lo -= pad; hi += pad;

      var Y = function (v) { return m.t + (hi - v) / (hi - lo) * ph; };
      var svg = svgRoot(W, H);
      var axisG = s('g', { 'aria-hidden': 'true' });
      var dataG = s('g');

      var ticks = axisTicks(lo, hi, 3);
      ticks.forEach(function (t) {
        var py = Y(t);
        if (py < m.t - 1 || py > m.t + ph + 1) return;
        axisG.appendChild(s('line', { x1: m.l, y1: py, x2: m.l + pw, y2: py, stroke: GRID, 'stroke-width': 1 }));
        axisG.appendChild(s('text', {
          x: m.l - 8, y: py + 4, fill: FAINT, 'font-family': MONO, 'font-size': 12,
          'text-anchor': 'end', text: t.toFixed(2)
        }));
      });

      var zeroY = Y(0);
      axisG.appendChild(s('line', { x1: m.l, y1: m.t, x2: m.l, y2: m.t + ph, stroke: AXIS, 'stroke-width': 1 }));
      axisG.appendChild(s('line', {
        x1: m.l, y1: zeroY, x2: m.l + pw, y2: zeroY,
        stroke: AXIS, 'stroke-width': 1
      }));

      var slotW = pw / cats.length;
      var groupW = Math.min(slotW * 0.66, 46);
      var barW = groupW / groups.length;
      var tip = null;

      cats.forEach(function (cat, ci) {
        var cx = m.l + slotW * (ci + 0.5);
        groups.forEach(function (se, gi) {
          var n = numOrNull(se.values[ci]);
          if (n === null) return;
          var y = Y(n);
          var x = cx - groupW / 2 + gi * barW;
          var top = Math.min(y, zeroY);
          var hgt = Math.max(1, Math.abs(y - zeroY));
          dataG.appendChild(s('rect', {
            x: (x + 1).toFixed(2), y: top.toFixed(2),
            width: Math.max(1, barW - 2).toFixed(2), height: hgt.toFixed(2),
            fill: se.color || SERIES[gi % SERIES.length], rx: 1
          }));
        });
        axisG.appendChild(s('text', {
          x: cx, y: m.t + ph + 18, fill: FAINT, 'font-family': MONO, 'font-size': 12,
          'text-anchor': 'middle', text: String(cat)
        }));
      });

      if (opts.xLabel) {
        svg.appendChild(s('text', {
          x: m.l + pw / 2, y: H - 6, fill: MUTED, 'font-family': UI, 'font-size': 12,
          'text-anchor': 'middle', text: opts.xLabel
        }));
      }

      svg.appendChild(axisG);
      svg.appendChild(dataG);
      host.replaceChildren(svg);
      void tip;

      var lg = legend(groups.map(function (se, gi) {
        return { label: String(se.name == null ? 'series ' + (gi + 1) : se.name), color: se.color || SERIES[gi % SERIES.length] };
      }));
      if (lg) host.appendChild(lg);
      svg.setAttribute('aria-label', opts.ariaLabel || 'Grouped amplitude plot.');
    });
  }

  global.QCharts = {
    roc: roc,
    hbars: hbars,
    grouped: grouped,
    fmt: fmt,
    fmtInt: fmtInt,
    isNum: isNum,
    numOrNull: numOrNull,
    SERIES: SERIES,
    INK: INK,
    MUTED: MUTED,
    FAINT: FAINT,
    MONO: MONO
  };
}(typeof window !== 'undefined' ? window : globalThis));

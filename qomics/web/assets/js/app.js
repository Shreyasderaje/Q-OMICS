/* =============================================================================
   Q-OMICS — app.js
   Vanilla, no build step, no external requests. The API base is same-origin
   by default (the FastAPI app serves this page from /); override with
   ?api=<base-url> or window.QOMICS_API_BASE for a split deployment.
   ========================================================================== */
(function (global) {
  'use strict';

  /* ------------------------------------------------------------------ setup */

  var doc = global.document;

  var API_BASE = (function () {
    var injected = global.QOMICS_API_BASE;
    if (typeof injected === 'string' && injected) return injected.replace(/\/+$/, '');
    try {
      var q = new URLSearchParams(global.location.search).get('api');
      if (q) return q.replace(/\/+$/, '');
    } catch (e) { /* URLSearchParams unavailable: fall through */ }
    return '';
  }());

  var REQUEST_TIMEOUT_MS = 20000;

  var state = {
    health: null,
    metrics: null,
    overview: null,
    architecture: null,
    features: null,
    quantum: null,
    cohort: null,
    biomarkers: null,
    selectedSymbol: null,
    bm: { block: 'all', limit: 50, query: '', sort: { key: 'rank', dir: 'asc' } },
    co: { limit: 60, offset: 0, stage: '', histology: '', query: '', sort: { key: 'sample_id', dir: 'asc' } },
    predictMode: 'structured',
    lastPrediction: null,
    stageChartKey: null
  };

  /* ------------------------------------------------------------ dom helpers */

  function h(tag, attrs, children) {
    var node = doc.createElement(tag);
    if (attrs) {
      for (var k in attrs) {
        if (!Object.prototype.hasOwnProperty.call(attrs, k)) continue;
        var v = attrs[k];
        if (v === null || v === undefined || v === false) continue;
        if (k === 'class') node.className = v;
        else if (k === 'text') node.textContent = String(v);
        else if (k.slice(0, 2) === 'on' && typeof v === 'function') node.addEventListener(k.slice(2), v);
        else if (v === true) node.setAttribute(k, '');
        else node.setAttribute(k, String(v));
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
      node.appendChild(c.nodeType ? c : doc.createTextNode(String(c)));
    }
  }

  function clear(node) { if (node) node.replaceChildren(); }

  function panel(name) { return doc.querySelector('[data-panel="' + name + '"]'); }
  function bind(name) { return doc.querySelector('[data-bind="' + name + '"]'); }
  function byId(id) { return doc.getElementById(id); }

  function setText(name, value) {
    var el = bind(name);
    if (el) el.textContent = value === null || value === undefined || value === '' ? '—' : String(value);
  }

  /* ---------------------------------------------------------- format helpers */

  function numOrNull(v) {
    if (v === null || v === undefined || v === '') return null;
    var n = typeof v === 'number' ? v : parseFloat(v);
    return (typeof n === 'number' && isFinite(n)) ? n : null;
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

  function fmtThousands(v) {
    var n = numOrNull(v);
    if (n === null) return '—';
    return Math.round(n).toLocaleString('en-US');
  }

  function fmtPct(v, dp) {
    var n = numOrNull(v);
    if (n === null) return '—';
    return (n * 100).toFixed(dp === undefined ? 1 : dp) + '%';
  }

  function fmtSigned(v, dp) {
    var n = numOrNull(v);
    if (n === null) return '—';
    return (n > 0 ? '+' : '') + n.toFixed(dp === undefined ? 2 : dp);
  }

  function fmtDelta(v, dp) {
    var n = numOrNull(v);
    if (n === null) return '—';
    return (n > 0 ? '+' : '') + n.toFixed(dp === undefined ? 3 : dp);
  }

  /* Deterministic ISO rendering — no timezone surprises. */
  function fmtStamp(iso) {
    if (typeof iso !== 'string' || !iso) return null;
    var m = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})/.exec(iso);
    if (!m) return iso;
    return m[1] + '-' + m[2] + '-' + m[3] + ' ' + m[4] + ':' + m[5] + 'Z';
  }

  function titleCase(s) {
    if (typeof s !== 'string' || !s) return '';
    return s.replace(/[_-]+/g, ' ').replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  function blockLabel(b) { return titleCase(b); }

  function directionLabel(d) {
    if (d === 'depleted_in_early_stage') return 'depleted in stage I';
    if (d === 'enriched_in_early_stage') return 'enriched in stage I';
    if (d === 'depleted_in_advanced_stage') return 'depleted in stage II–IV';
    if (d === 'enriched_in_advanced_stage') return 'enriched in stage II–IV';
    return titleCase(d);
  }

  function stageClass(stage) {
    var s0 = String(stage == null ? '' : stage).toUpperCase().trim();
    if (/^(I|IA|IB|IC)$/.test(s0)) return 'early';
    if (/^(II|III|IV)/.test(s0)) return 'advanced';
    return 'neutral';
  }

  function deltaClass(v) {
    var n = numOrNull(v);
    if (n === null || n === 0) return '';
    return n > 0 ? 'up' : 'down';
  }

  /* ------------------------------------------------------------ api client */

  function ApiError(message, status, detail, url) {
    this.name = 'ApiError';
    this.message = message;
    this.status = status === undefined ? null : status;
    this.detail = detail === undefined ? null : detail;
    this.url = url || null;
  }
  ApiError.prototype = Object.create(Error.prototype);
  ApiError.prototype.constructor = ApiError;

  function url(path, params) {
    var out = API_BASE + path;
    if (params) {
      var qs = [];
      for (var k in params) {
        if (!Object.prototype.hasOwnProperty.call(params, k)) continue;
        if (params[k] === null || params[k] === undefined || params[k] === '') continue;
        qs.push(encodeURIComponent(k) + '=' + encodeURIComponent(String(params[k])));
      }
      if (qs.length) out += (out.indexOf('?') === -1 ? '?' : '&') + qs.join('&');
    }
    return out;
  }

  function withTimeout() {
    if (typeof AbortController !== 'function') return { signal: null, clear: function () {} };
    var ctrl = new AbortController();
    var timer = global.setTimeout(function () { ctrl.abort(); }, REQUEST_TIMEOUT_MS);
    return { signal: ctrl.signal, clear: function () { global.clearTimeout(timer); } };
  }

  function readError(res) {
    return res.text().then(function (text) {
      var detail = null;
      if (text) {
        try {
          var body = JSON.parse(text);
          if (body && typeof body.detail === 'string') detail = body.detail;
          else if (body && Array.isArray(body.detail) && body.detail.length) {
            detail = body.detail.map(function (d) {
              return d && d.msg ? String(d.msg) : JSON.stringify(d);
            }).join('; ');
          } else if (typeof body === 'string') detail = body;
        } catch (e) { detail = text.slice(0, 400); }
      }
      return detail;
    });
  }

  function request(path, options) {
    options = options || {};
    var t = withTimeout();
    var init = { method: options.method || 'GET', signal: t.signal, headers: { Accept: 'application/json' } };
    if (options.body !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(options.body);
    }
    if (options.headers) {
      for (var hk in options.headers) {
        if (Object.prototype.hasOwnProperty.call(options.headers, hk)) init.headers[hk] = options.headers[hk];
      }
    }
    return global.fetch(url(path, options.params), init).then(function (res) {
      t.clear();
      if (!res.ok) {
        return readError(res).then(function (detail) {
          throw new ApiError('HTTP ' + res.status + ' for ' + path, res.status, detail, path);
        });
      }
      return res.text().then(function (text) {
        if (!text) return null;
        try { return JSON.parse(text); }
        catch (e) { throw new ApiError('Malformed JSON from ' + path, res.status, text.slice(0, 200), path); }
      });
    }, function (err) {
      t.clear();
      if (err && err.name === 'ApiError') throw err;
      if (err && err.name === 'AbortError') {
        throw new ApiError('Request timed out after ' + (REQUEST_TIMEOUT_MS / 1000) + 's for ' + path, 0, 'timeout', path);
      }
      throw new ApiError('Network error for ' + path, 0, (err && err.message) || String(err), path);
    });
  }

  var api = {
    health: function () { return request('/api/health'); },
    overview: function () { return request('/api/overview'); },
    metrics: function () { return request('/api/metrics'); },
    biomarkers: function (params) { return request('/api/biomarkers', { params: params }); },
    biomarker: function (symbol) { return request('/api/biomarkers/' + encodeURIComponent(symbol)); },
    quantum: function () { return request('/api/quantum'); },
    predict: function (body) { return request('/api/predict', { method: 'POST', body: body }); },
    cohort: function (params) { return request('/api/cohort', { params: params }); },
    cohortCsvUrl: function () { return url('/api/cohort/download', { format: 'csv' }); },
    cohortDownload: function () {
      return global.fetch(url('/api/cohort/download', { format: 'csv' }), {
        headers: { Accept: 'text/csv' }
      }).then(function (res) {
        if (!res.ok) {
          return readError(res).then(function (detail) {
            throw new ApiError('HTTP ' + res.status + ' for /api/cohort/download', res.status, detail, '/api/cohort/download');
          });
        }
        return res.blob();
      });
    },
    architecture: function () { return request('/api/architecture'); },
    features: function () { return request('/api/features'); }
  };

  /* -------------------------------------------------------- panel UI states */

  function setLoading(el, lines) {
    if (!el) return;
    var n = lines || 4;
    var kids = [];
    for (var i = 0; i < n; i++) kids.push(h('div', { class: 'skel-line' }));
    el.replaceChildren(h('div', { class: 'skeleton', 'aria-busy': 'true', 'aria-live': 'polite' }, kids));
  }

  function setBusy(el, message) {
    if (!el) return;
    el.replaceChildren(h('div', { class: 'state-inline', 'aria-busy': 'true' }, [
      h('span', { class: 'spinner', 'aria-hidden': 'true' }),
      h('span', { text: message || 'Loading' })
    ]));
  }

  function describeError(err) {
    if (err && err.status === 503) {
      return {
        title: 'Model artefacts not present',
        body: 'The server is running but the trained model artefacts could not be loaded. Train or restore the checkpoint, then restart the server.',
        code: (err.detail || 'model not trained; run scripts/train.py') + (err.url ? '  —  ' + err.url : '')
      };
    }
    if (err && err.status === 404) {
      return {
        title: 'Endpoint not found',
        body: 'The server responded 404 for this route. The interface expects the routes defined in docs/API_CONTRACT.md; the running server appears to be a different version.',
        code: err.url || ''
      };
    }
    if (err && err.status === 422) {
      return {
        title: 'Request rejected',
        body: 'The server could not validate the request body. Check the payload shape against the contract.',
        code: err.detail || ''
      };
    }
    if (err && err.status === 0) {
      return {
        title: 'Cannot reach the API',
        body: 'No response from the server. Start it with uvicorn qomics.serving.app:app, or python scripts/serve.py, then reload.',
        code: (err.detail || '') + (err.url ? '  —  ' + err.url : '')
      };
    }
    return {
      title: 'Request failed',
      body: 'The panel could not be populated.',
      code: (err && err.message ? err.message : String(err))
    };
  }

  function setError(el, err, context) {
    if (!el) return;
    var d = describeError(err);
    el.replaceChildren(h('div', { class: 'state state--error', role: 'alert' }, [
      h('p', { class: 'state-title', text: context ? context + ' — ' + d.title : d.title }),
      h('p', { text: d.body }),
      d.code ? h('p', { class: 'state-code', text: d.code }) : null
    ]));
  }

  function setEmpty(el, message) {
    if (!el) return;
    el.replaceChildren(h('div', { class: 'state state--empty', role: 'status' }, [
      h('p', { class: 'state-title', text: 'No records' }),
      h('p', { text: message || 'The endpoint responded successfully with an empty result set.' })
    ]));
  }

  /* ------------------------------------------------------------ data tables */

  function compare(a, b) {
    var na = numOrNull(a), nb = numOrNull(b);
    if (na !== null && nb !== null) return na - nb;
    if (a === null || a === undefined || a === '') return 1;
    if (b === null || b === undefined || b === '') return -1;
    return String(a).toLowerCase().localeCompare(String(b).toLowerCase());
  }

  /**
   * Generic dense table. `columns` = [{key, label, align, sortValue, render, sortable}]
   * Rendering always goes through createElement/textContent.
   */
  function dataTable(cfg) {
    var columns = cfg.columns || [];
    var rows = (cfg.rows || []).slice();
    var sort = cfg.sort || null;
    var onSort = cfg.onSort || null;

    if (sort && sort.key) {
      var col = columns.filter(function (c) { return c.key === sort.key; })[0];
      if (col) {
        var dir = sort.dir === 'desc' ? -1 : 1;
        rows.sort(function (r1, r2) {
          return dir * compare(col.sortValue ? col.sortValue(r1) : r1[col.key], col.sortValue ? col.sortValue(r2) : r2[col.key]);
        });
      }
    }

    var thead = h('thead', null, h('tr', null, columns.map(function (c) {
      var isSorted = sort && sort.key === c.key;
      var thAttrs = { scope: 'col' };
      if (c.align === 'left') thAttrs['class'] = 'left';
      else thAttrs['class'] = 'num';
      if (isSorted) thAttrs['aria-sort'] = sort.dir === 'asc' ? 'ascending' : 'descending';
      if (c.sortable) {
        thAttrs['class'] += ' sortable';
        thAttrs.tabindex = '0';
        thAttrs.role = 'columnheader';
        var activate = function () {
          if (onSort) onSort(c.key, isSorted && sort.dir === 'asc' ? 'desc' : 'asc');
        };
        thAttrs['aria-label'] = 'Sort by ' + c.label;
        thAttrs.onclick = activate;
        thAttrs.onkeydown = function (ev) {
          if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); activate(); }
        };
      }
      return h('th', thAttrs, [
        c.label,
        c.sortable ? h('span', { class: 'sort-ind', 'aria-hidden': 'true', text: isSorted ? (sort.dir === 'asc' ? '^' : 'v') : '·' }) : null
      ]);
    })));

    var tbody = h('tbody');
    rows.forEach(function (row, i) {
      var tr = h('tr', { 'data-index': i });
      if (cfg.rowClass) {
        var cls = cfg.rowClass(row);
        if (cls) tr.className = cls;
      }
      columns.forEach(function (c) {
        var tdAttrs = c.align === 'left' ? { class: 'text' } : { class: 'num' };
        if (c.nowrap === false) tdAttrs.className += ' wrap';
        var content = c.render ? c.render(row, i) : row[c.key];
        tr.appendChild(h('td', tdAttrs, content === null || content === undefined || content === '' ? '—' : content));
      });
      if (cfg.onRowClick) {
        tr.classList.add('row-clickable');
        tr.setAttribute('tabindex', '0');
        var fire = function () { cfg.onRowClick(row, tr); };
        tr.addEventListener('click', fire);
        tr.addEventListener('keydown', function (ev) {
          if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); fire(); }
        });
      }
      tbody.appendChild(tr);
    });

    var table = h('table', { class: 'data' }, [
      cfg.caption ? h('caption', { text: cfg.caption }) : null,
      thead,
      tbody
    ]);
    return table;
  }

  function bar(value, max, cls) {
    var v = numOrNull(value);
    var m = numOrNull(max);
    if (v === null) return h('span', { class: 'num', text: '—' });
    var frac = (m && m > 0) ? Math.max(0, Math.min(1, Math.abs(v) / m)) : 0;
    return h('span', { class: 'cell-bar' }, [
      h('span', { class: 'bar-track', 'aria-hidden': 'true' },
        h('span', { class: 'bar-fill' + (cls ? ' ' + cls : ''), style: 'width:' + (frac * 100).toFixed(1) + '%' })),
      h('span', { text: fmt(v, 4) })
    ]);
  }

  /* ============================================================== /api/health */

  function renderHealthStatus(health) {
    state.health = health;
    var dot = bind('status-dot');
    var text = bind('status-text');
    var loaded = health && health.model_loaded === true;
    var mode = loaded ? 'ok' : 'degraded';

    if (dot) {
      dot.setAttribute('data-state', mode);
      dot.setAttribute('title', loaded ? 'Model loaded' : 'Server up, model not loaded');
    }
    if (text) {
      text.textContent = loaded
        ? 'online · ' + (health.version || 'unversioned')
        : 'degraded · artefacts missing';
    }
    setText('version', health && health.version ? 'v' + health.version : 'v—');
    setText('footer-version', health && health.version ? health.version : null);
    setText('footer-loaded', health ? (loaded ? 'loaded' : 'not loaded') : null);
    setText('footer-cohort', health ? health.cohort : null);
    setText('footer-uptime', health && numOrNull(health.runtime_seconds) !== null
      ? numOrNull(health.runtime_seconds).toFixed(1) + ' s' : null);
    setText('spec-trained', health ? fmtStamp(health.trained_at) : null);
  }

  function showGate(reason, title) {
    var gate = byId('connection-gate');
    if (!gate) return;
    gate.hidden = false;
    var heading = gate.querySelector('.gate-title');
    if (heading && title) heading.textContent = title;
    var bodyEl = gate.querySelector('.gate-body');
    if (bodyEl && title === 'Model artefacts not present') {
      bodyEl.textContent = 'The server responded, but the trained model artefacts could not be loaded. ' +
        'Panels that depend on the checkpoint will report 503 until scripts/train.py has been run. ' +
        'The commands below start the server once artefacts are present.';
    }
    var detail = bind('gate-detail');
    if (detail) {
      detail.textContent = reason
        ? 'Last attempt: ' + reason
        : 'Endpoint checked: GET ' + (API_BASE || '') + '/api/health';
    }
  }

  function hideGate() {
    var gate = byId('connection-gate');
    if (gate) gate.hidden = true;
  }

  function notice(text) {
    var existing = byId('runtime-notice');
    if (existing) existing.remove();
    if (!text) return;
    var el = h('div', { class: 'gate', id: 'runtime-notice' },
      h('div', { class: 'wrap' },
        h('div', { class: 'gate-inner' },
          h('div', { class: 'gate-head' },
            h('span', { class: 'eyebrow', text: 'Runtime' }),
            h('h2', { class: 'gate-title', text: text.title }),
            h('p', { class: 'gate-body', text: text.body })))));
    var main = byId('main');
    if (main && main.parentNode) main.parentNode.insertBefore(el, main);
  }

  /* ============================================================= /api/overview */

  function renderOverview(ov) {
    state.overview = ov;
    if (!ov) return;

    var product = ov.product || {};
    var cohort = ov.cohort || {};
    var head = ov.headline_metrics || {};
    var shapes = cohort.block_shapes || {};

    if (product.tagline) {
      setText('hero-sub', product.tagline);
      setText('hero-eyebrow', [product.name, cohort.study, 'research use only']
        .filter(Boolean).join(' · '));
    }
    setText('hero-cohort', cohort.study || null);
    setText('hero-target', cohort.target_description || cohort.target || null);

    /* cohort composition panel */
    var body = panel('overview-cohort');
    if (body) {
      var rows = [];
      rows.push(specRow('Study', cohort.study));
      rows.push(specRow('Study id', cohort.study_id));
      rows.push(specRow('Samples', cohort.n_samples === undefined ? null : fmtThousands(cohort.n_samples)));
      rows.push(specRow('Patients', cohort.n_patients === undefined ? null : fmtThousands(cohort.n_patients)));
      rows.push(specRow('Target', cohort.target_description || cohort.target));
      var cc = cohort.class_counts || {};
      rows.push(specRow('Class counts', Object.keys(cc).map(function (k) {
        return k + ' ' + fmtThousands(cc[k]);
      }).join('  ·  ') || null));
      rows.push(specRow('Class balance', fmt(cohort.class_balance, 4)));
      var shapeText = Object.keys(shapes).map(function (k) {
        return blockLabel(k) + ' ' + fmtThousands(shapes[k]);
      }).join('  ·  ');
      rows.push(specRow('Block widths', shapeText || null));
      clear(body);
      body.appendChild(h('dl', { class: 'spec-list' }, rows));
    }

    var totalFeatures = Object.keys(shapes).reduce(function (acc, k) {
      return acc + (numOrNull(shapes[k]) || 0);
    }, 0);

    if (numOrNull(head.hybrid_auc_mean) !== null) {
      setText('ds-auc', fmt(head.hybrid_auc_mean, 4));
      var aucNote = '± ' + fmt(head.hybrid_auc_std, 4) + ' over ' + fmtInt(head.n_splits) + ' splits';
      if (numOrNull(head.improvement_over_best_classical) !== null && head.best_baseline_name) {
        aucNote += ' · ' + fmtDelta(head.improvement_over_best_classical, 4) + ' vs ' + head.best_baseline_name;
      }
      setText('ds-auc-note', aucNote);
    }
    setText('spec-splits', numOrNull(head.n_splits) !== null
      ? fmtInt(head.n_splits) + ' repeated stratified' : null);
    if (numOrNull(cohort.n_samples) !== null) {
      setText('ds-n', fmtThousands(cohort.n_samples));
      setText('ds-n-note', cohort.study || '');
    }
    if (totalFeatures > 0) {
      setText('ds-features', fmtThousands(totalFeatures));
      setText('ds-features-note', ['transcriptome', 'genome', 'proteome']
        .filter(function (b) { return numOrNull(shapes[b]) !== null; })
        .map(function (b) { return fmtThousands(shapes[b]) + ' ' + blockLabel(b).toLowerCase(); })
        .join(' / '));
    }

    var host = byId('chart-modality');
    if (host) {
      var contrib = (ov.modality_contribution || []).map(function (c) {
        return { label: blockLabel(c.block), value: numOrNull(c.importance), valueText: fmt(c.importance, 3) };
      });
      // The split count is stated in prose, so it has to track the run rather
      // than sit in the markup as a constant that goes stale on re-training.
      setText('modality-splits',
        (ov.headline_metrics && ov.headline_metrics.n_splits)
          ? (ov.headline_metrics.n_splits + ' repeated patient-disjoint')
          : 'the cross-validation');
      if (contrib.length) {
        QCharts.hbars(host, contrib, { axisLabel: 'mean attention-rollout importance' });
      } else {
        QCharts.hbars(host, [], { emptyMessage: 'No modality contribution data in /api/overview.' });
      }
    }

    var bs = panel('block-shapes');
    if (bs) {
      var keys = Object.keys(shapes);
      if (!keys.length) {
        setEmpty(bs, 'No block shapes returned in /api/overview.');
      } else {
        clear(bs);
        bs.appendChild(h('dl', { class: 'spec-list' }, keys.map(function (k) {
          return specRow(blockLabel(k), fmtThousands(shapes[k]));
        }).concat([specRow('Total', fmtThousands(totalFeatures))])));
      }
    }

    var strip = panel('datastrip');
    if (strip) strip.setAttribute('data-loaded', '1');
  }

  function specRow(k, v) {
    return h('div', { class: 'spec-row' }, [h('dt', { text: k }), h('dd', { text: v === null || v === undefined || v === '' ? '—' : String(v) })]);
  }

  /* ========================================================== /api/metrics */

  var METRIC_COLUMNS = [
    { key: 'name', label: 'Model', align: 'left', nowrap: false, sortable: true },
    { key: 'family', label: 'Family', align: 'left', sortable: true, render: function (r) { return titleCase(r.family); } },
    { key: 'auc_mean', label: 'AUC', sortable: true, render: function (r) { return fmt(r.auc_mean, 4); } },
    { key: 'auc_ci', label: '95% CI', render: function (r) {
        var lo = numOrNull(r.auc_ci_low), hi = numOrNull(r.auc_ci_high);
        if (lo === null || hi === null) return '—';
        return lo.toFixed(3) + '–' + hi.toFixed(3);
      } },
    { key: 'pr_auc_mean', label: 'PR AUC', sortable: true, render: function (r) { return fmt(r.pr_auc_mean, 4); } },
    { key: 'f1_mean', label: 'F1', sortable: true, render: function (r) { return fmt(r.f1_mean, 4); } },
    { key: 'sensitivity_mean', label: 'Sens.', sortable: true, render: function (r) { return fmt(r.sensitivity_mean, 4); } },
    { key: 'specificity_mean', label: 'Spec.', sortable: true, render: function (r) { return fmt(r.specificity_mean, 4); } },
    { key: 'brier_mean', label: 'Brier', sortable: true, render: function (r) { return fmt(r.brier_mean, 4); } },
    { key: 'folds', label: 'Folds', render: function (r) {
        var f = Array.isArray(r.folds) ? r.folds : [];
        if (!f.length) return '—';
        var aucs = f.map(function (x) { return numOrNull(x.auc); }).filter(function (x) { return x !== null; });
        if (!aucs.length) return String(f.length);
        var mn = Math.min.apply(null, aucs), mx = Math.max.apply(null, aucs);
        return String(f.length) + '  (' + mn.toFixed(3) + '–' + mx.toFixed(3) + ')';
      } },
    { key: 'seconds_mean', label: 's/sample', sortable: true, render: function (r) { return fmt(r.seconds_mean, 1); } }
  ];

  function renderMetrics(m) {
    state.metrics = m;
    if (!m) return;

    setText('results-lede', m.target
      ? m.target + ' · ' + fmtInt(m.n_splits) + ' repeated stratified splits.'
      : 'Target definition is served by /api/metrics.');

    var primary = m.primary || '';
    var models = Array.isArray(m.models) ? m.models : [];
    var primaryName = resolvePrimaryName(models, primary);

    var chip = bind('primary-chip');
    if (chip) {
      chip.textContent = 'primary ' + (primary || '—') +
        (primaryName && primaryName !== primary ? '  ·  ' + primaryName : '');
    }

    var host = panel('metrics-table');
    if (host) {
      if (!models.length) {
        setEmpty(host, '/api/metrics returned no model rows.');
      } else {
        var table = dataTable({
          columns: METRIC_COLUMNS,
          rows: models,
          sort: { key: 'auc_mean', dir: 'desc' },
          caption: 'Mean and standard deviation across folds. Dispersion in the Folds column is the min–max out-of-fold AUC.',
          rowClass: function (r) { return r.name === primaryName ? 'is-primary' : ''; }
        });
        clear(host);
        host.appendChild(table);
      }
    }

    var sigHost = panel('significance');
    if (sigHost) {
      var sigs = Array.isArray(m.significance) ? m.significance : [];
      if (!sigs.length) {
        setEmpty(sigHost, 'No significance comparisons were returned.');
      } else {
        clear(sigHost);
        sigHost.appendChild(h('div', null, sigs.map(function (s) {
          return h('div', { class: 'sig-row' }, [
            h('p', { class: 'sig-name', text: String(s.comparison == null ? '—' : s.comparison) }),
            h('span', {
              class: 'sig-verdict',
              'data-sig': s['significant_at_0.05'] ? '1' : '0',
              text: s['significant_at_0.05'] ? 'significant at 0.05' : 'not significant at 0.05'
            }),
            h('span', { class: 'sig-stats' }, [
              h('span', { text: 'Δ AUC ' + fmtDelta(s.mean_delta, 4) }),
              h('span', { text: 't ' + fmt(s.t_statistic, 2) }),
              h('span', { text: 'p ' + fmtP(s.p_value) })
            ])
          ]);
        })));
      }
    }

    var rocHost = byId('chart-roc');
    if (rocHost) {
      var rocData = m.roc_curve || {};
      QCharts.roc(rocHost, rocData, { primaryName: primaryName });
    }
  }

  /**
   * /api/metrics reports `primary` as a model id (e.g. "hybrid_transformer") while
   * the rows carry `name` and `family`. Resolve the id to a row by exact name,
   * then by family, then by slug-token prefix on the name.
   */
  function resolvePrimaryName(models, primary) {
    if (!primary || !models.length) return '';
    var i, k, m, hit;
    for (i = 0; i < models.length; i++) if (models[i].name === primary) return models[i].name;
    for (i = 0; i < models.length; i++) if (models[i].family && models[i].family === primary) return models[i].name;
    var tokens = String(primary).toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
    if (!tokens.length) return '';
    for (i = 0; i < models.length; i++) {
      m = String(models[i].name || '').toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
      hit = true;
      for (k = 0; k < tokens.length; k++) { if (m[k] !== tokens[k]) { hit = false; break; } }
      if (hit) return models[i].name;
    }
    return '';
  }

  function fmtP(p) {
    var n = numOrNull(p);
    if (n === null) return '—';
    if (n < 0.001) return '< 0.001';
    return n.toFixed(4);
  }

  /* ====================================================== /api/architecture */

  function renderArchitecture(a) {
    if (!a) return;
    var host = panel('pipeline');
    var stages = Array.isArray(a.stages) ? a.stages : [];

    if (host) {
      if (!stages.length) {
        setEmpty(host, 'No pipeline stages were returned by /api/architecture.');
      } else {
        clear(host);
        stages.forEach(function (st, i) {
          var tokens = numOrNull(st.tokens);
          host.appendChild(h('li', { class: 'pipeline-stage' }, [
            h('span', { class: 'pipe-index', text: String(i + 1).padStart(2, '0') }),
            h('h3', { class: 'pipe-title', text: String(st.title == null ? st.id : st.title) }),
            h('p', { class: 'pipe-detail', text: String(st.detail == null ? '' : st.detail) }),
            h('p', { class: 'pipe-tokens' }, [
              tokens === null || tokens === 0
                ? h('span', { class: 'zero', text: 'no token output' })
                : h('span', { text: fmtThousands(tokens) + ' tokens' })
            ])
          ]));
        });
      }
    }

    var m = a.model || {};
    var mg = panel('model-graph');
    if (mg) {
      clear(mg);
      mg.appendChild(h('dl', { class: 'spec-list' }, [
        specRow('Parameters', fmtThousands(m.parameters)),
        specRow('d_model', fmtInt(m.d_model)),
        specRow('Attention heads', fmtInt(m.n_heads)),
        specRow('Layers', fmtInt(m.n_layers)),
        specRow('Feed-forward', fmtInt(m.dim_feedforward)),
        specRow('Dropout', fmt(m.dropout, 2))
      ]));
    }

    var t = a.training || {};
    var tr = panel('training');
    if (tr) {
      clear(tr);
      tr.appendChild(h('dl', { class: 'spec-list' }, [
        specRow('Epochs', fmtInt(t.epochs)),
        specRow('Batch size', fmtInt(t.batch_size)),
        specRow('Learning rate', fmt(t.learning_rate, 5)),
        specRow('Weight decay', fmt(t.weight_decay, 5)),
        specRow('Loss', t.loss),
        specRow('Splits', fmtInt(t.n_splits))
      ]));
    }
  }

  /* ============================================================ /api/features */

  function renderFeatures(f) {
    var host = panel('features');
    if (numOrNull(f && f.n_total) !== null) {
      setText('spec-tokens', fmtThousands(f.n_total) + ' tokens');
      setText('ds-features', fmtThousands(f.n_total));
    }
    if (!host) return;
    var blocks = f && Array.isArray(f.blocks) ? f.blocks : [];
    if (!blocks.length) {
      setEmpty(host, 'No feature inventory returned.');
      return;
    }

    var total = numOrNull(f.n_total);
    var rows = blocks.map(function (b) {
      var axes = Array.isArray(b.axes) ? b.axes : [];
      var top = axes.slice().sort(function (x, y) { return (numOrNull(y.count) || 0) - (numOrNull(x.count) || 0); }).slice(0, 4);
      return {
        block: b.block,
        n: numOrNull(b.n_features),
        axesCount: axes.length,
        top: top,
        topText: top.length
          ? top.map(function (a) { return a.axis + ' (' + fmtInt(a.count) + ')'; }).join('  ·  ')
          : 'no pathway axis assigned'
      };
    });

    var table = dataTable({
      columns: [
        { key: 'block', label: 'Block', align: 'left', sortable: true, render: function (r) { return blockLabel(r.block); } },
        { key: 'n', label: 'Features', sortable: true, render: function (r) { return fmtThousands(r.n); } },
        { key: 'axesCount', label: 'Axes', sortable: true, render: function (r) { return fmtInt(r.axesCount); } },
        { key: 'share', label: 'Share of total', render: function (r) {
            if (total === null || !total || r.n === null) return '—';
            return fmtPct(r.n / total, 1);
          } },
        { key: 'top', label: 'Largest axes', align: 'left', nowrap: false, render: function (r) {
            return h('span', { class: 'axes-list' }, r.top.map(function (a) {
              return h('span', { class: 'chip', text: a.axis + ' · ' + fmtInt(a.count) });
            }).concat(r.top.length ? [] : [h('span', { class: 'chip chip-muted', text: 'unannotated' })]));
          } }
      ],
      rows: rows,
      sort: { key: 'n', dir: 'desc' },
      caption: 'Feature inventory behind reduction, kernel encoding and tokenisation. Total: ' + fmtThousands(total) + ' features.'
    });
    clear(host);
    host.appendChild(table);
  }

  /* ============================================================= /api/quantum */

  function renderQuantum(q) {
    if (!q) return;
    setText('spec-backend', q.backend);
    setText('spec-kernel', 'fidelity · ' + fmtInt(q.n_qubits) + '-qubit');
    setText('spec-map', q.feature_map + (q.n_layers ? ' · ' + fmtInt(q.n_layers) + ' layers' : ''));
    setText('spec-kernel-detail', q.normalisation);
    if (numOrNull(q.n_qubits) !== null) {
      setText('ds-qubits', fmtInt(q.n_qubits));
      setText('ds-qubits-note', 'Hilbert dimension ' + fmtThousands(q.hilbert_dimension) + ' · ' + String(q.feature_map || ''));
    }

    var reg = panel('quantum-register');
    if (reg) {
      clear(reg);
      reg.appendChild(h('dl', { class: 'spec-list' }, [
        specRow('Simulator', q.backend),
        specRow('Qubits', fmtInt(q.n_qubits)),
        specRow('Hilbert dimension', fmtThousands(q.hilbert_dimension)),
        specRow('Feature map', q.feature_map),
        specRow('Reupload', q.reupload === true ? 'yes' : (q.reupload === false ? 'no' : '—')),
        specRow('Layers', fmtInt(q.n_layers)),
        specRow('Normalisation', q.normalisation),
        specRow('Encoding time', numOrNull(q.encoding_seconds) !== null ? fmt(q.encoding_seconds, 1) + ' s' : null)
      ]));
    }

    var kd = q.kernel_diagnostics || {};
    var diag = panel('quantum-diagnostics');
    if (diag) {
      clear(diag);
      diag.appendChild(h('dl', { class: 'spec-list' }, [
        specRow('Mean entry', fmt(kd.mean, 4)),
        specRow('Std entry', fmt(kd.std, 4)),
        specRow('Min / max', fmt(kd.min, 4) + '  /  ' + fmt(kd.max, 4)),
        specRow('Diagonal', fmt(kd.mean_diagonal, 4)),
        specRow('Condition number', fmt(kd.condition_number, 2)),
        specRow('Effective rank', fmt(kd.effective_rank, 1))
      ]));
    }

    var sep = q.labelled_separation || {};
    var sepHost = byId('chart-separation');
    if (sepHost) {
      var items = [];
      if (numOrNull(sep.quantum) !== null) items.push({ label: 'Quantum kernel', value: sep.quantum, valueText: fmt(sep.quantum, 4), color: QCharts.SERIES[0] });
      if (numOrNull(sep.rbf) !== null) items.push({ label: 'RBF kernel', value: sep.rbf, valueText: fmt(sep.rbf, 4), color: QCharts.SERIES[3] });
      if (numOrNull(sep.cosine) !== null) items.push({ label: 'Cosine kernel', value: sep.cosine, valueText: fmt(sep.cosine, 4), color: QCharts.SERIES[4] });
      QCharts.hbars(sepHost, items, {
        axisLabel: 'mean between-class distance minus within-class distance',
        emptyMessage: 'No labelled separation values returned.'
      });
    }

    var geo = q.geometric_difference || {};
    var geoHost = panel('quantum-geometry');
    if (geoHost) {
      var gRows = [];
      if (numOrNull(geo.g_quantum_vs_rbf) !== null) {
        gRows.push(geoFigure('g quantum vs RBF', geo.g_quantum_vs_rbf, geo.g_quantum_vs_rbf > 1));
      }
      if (numOrNull(geo.g_quantum_vs_cosine) !== null) {
        gRows.push(geoFigure('g quantum vs cosine', geo.g_quantum_vs_cosine, geo.g_quantum_vs_cosine > 1));
      }
      clear(geoHost);
      if (!gRows.length) {
        setEmpty(geoHost, 'No geometric-difference values returned.');
      } else {
        geoHost.appendChild(h('div', null, gRows));
        geoHost.appendChild(h('p', { class: 'geo-note', text: String(geo.interpretation || '') }));
        geoHost.appendChild(h('p', { class: 'geo-note' },
          'These are measured ratios, not an advantage claim. A value marginally above 1 means the ' +
          'fidelity geometry is distinguishable from the classical family it was compared against, ' +
          'not that it is better; the separation gap above is the quantity that moves the metric.'));
      }
    }

    var sv = q.state_vector_visualisation || {};
    var svHost = byId('chart-statevector');
    if (svHost) {
      var labels = Array.isArray(sv.labels) ? sv.labels : [];
      var amps = Array.isArray(sv.amplitudes) ? sv.amplitudes : [];
      var series = amps.filter(function (row) { return Array.isArray(row); }).map(function (row, i) {
        return {
          name: labels[i] || 'example ' + (i + 1),
          values: row,
          color: QCharts.SERIES[i % QCharts.SERIES.length]
        };
      });
      var nAmp = series.length ? series[0].values.length : 0;
      QCharts.grouped(svHost, {
        series: series,
        labels: Array.from({ length: nAmp }, function (_, i) { return 'a' + i; })
      }, {
        xLabel: 'amplitude index',
        ariaLabel: 'First ' + nAmp + ' state-vector amplitudes for ' + series.length + ' example' + (series.length === 1 ? '' : 's'),
        emptyMessage: 'No state-vector amplitudes returned.'
      });
    }
  }

  function geoFigure(name, value, above) {
    return h('div', null, [
      h('div', { class: 'geo-figure' }, [
        h('span', { class: 'geo-name', text: name }),
        h('span', { class: 'geo-val', text: fmt(value, 4) })
      ]),
      h('p', { class: 'geo-verdict' }, [
        h('span', {
          class: 'chip ' + (above ? 'chip-accent' : 'chip-muted'),
          text: above ? 'g > 1 — not classically reproducible' : 'g ≤ 1 — within the classical family'
        })
      ])
    ]);
  }

  /* ========================================================== /api/biomarkers */

  function renderBiomarkers(payload) {
    state.biomarkers = payload;
    var host = panel('biomarkers-table');
    if (!host) return;

    var list = payload && Array.isArray(payload.biomarkers) ? payload.biomarkers : [];
    if (!list.length) {
      setEmpty(host, 'No biomarkers returned for this filter.');
      updateBiomarkerStatus(payload);
      return;
    }

    var q = state.bm.query.trim().toLowerCase();
    var visible = q ? list.filter(function (b) {
      return [b.symbol, b.name, b.axis, b.block, b.modality, b.direction]
        .some(function (v) { return v !== null && v !== undefined && String(v).toLowerCase().indexOf(q) !== -1; });
    }) : list;

    if (!visible.length) {
      setEmpty(host, 'The server returned ' + list.length + ' rows, none matching "' + state.bm.query.trim() + '".');
      updateBiomarkerStatus(payload);
      return;
    }

    var maxNorm = visible.reduce(function (acc, b) {
      var v = numOrNull(b.importance_normalised);
      if (v === null) v = numOrNull(b.importance) || 0;
      return Math.max(acc, v);
    }, 0);

    var table = dataTable({
      columns: [
        { key: 'rank', label: '#', sortable: true, render: function (r) { return fmtInt(r.rank); } },
        { key: 'symbol', label: 'Symbol', align: 'left', sortable: true, render: function (r) {
            return h('span', { text: String(r.symbol == null ? r.name : r.symbol) });
          } },
        { key: 'block', label: 'Block', align: 'left', sortable: true, render: function (r) {
            return h('span', { class: 'chip', text: blockLabel(r.block) });
          } },
        { key: 'axis', label: 'Axis', align: 'left', sortable: true, render: function (r) {
            return r.axis ? String(r.axis) : h('span', { class: 'chip chip-muted', text: 'unannotated' });
          } },
        { key: 'importance', label: 'Importance', sortable: true, render: function (r) {
            var v = numOrNull(r.importance_normalised);
            return bar(v === null ? r.importance : v, maxNorm, 'neutral');
          } },
        { key: 'stability', label: 'Stability', sortable: true, render: function (r) { return fmt(r.stability, 3); } },
        { key: 'effect_size', label: "Cohen's d", sortable: true, render: function (r) {
            var v = numOrNull(r.effect_size);
            return h('span', { class: 'delta ' + deltaClass(v), text: fmtSigned(v, 2) });
          } },
        { key: 'q_value', label: 'FDR q', sortable: true, render: function (r) {
            var sig = r.fdr_significant === true;
            return h('span', {
              class: 'chip ' + (sig ? 'chip-accent' : 'chip-muted'),
              text: fmtP(r.q_value),
              title: sig
                ? 'Survives Benjamini-Hochberg correction at q < 0.05'
                : 'Does not survive multiple-testing correction'
            });
          } },
        { key: 'mean_early', label: 'Stage I', sortable: true, render: function (r) { return fmtSigned(r.mean_early, 2); } },
        { key: 'mean_advanced', label: 'Stage II–IV', sortable: true, render: function (r) { return fmtSigned(r.mean_advanced, 2); } },
        { key: 'direction', label: 'Direction', align: 'left', sortable: true, render: function (r) {
            var d = String(r.direction || '');
            var cls = d.indexOf('early_stage') !== -1 ? 'chip-early' : (d ? 'chip-advanced' : 'chip-muted');
            return h('span', { class: 'chip ' + cls, text: directionLabel(r.direction) });
          } }
      ],
      rows: visible,
      sort: state.bm.sort,
      onSort: function (key, dir) { state.bm.sort = { key: key, dir: dir }; renderBiomarkers(state.biomarkers); },
      caption: 'Method: ' + (payload.method || 'unspecified') + '. Importance is the mean attention-rollout weight.',
      rowClass: function (r) { return String(r.symbol) === state.selectedSymbol ? 'is-primary' : ''; },
      onRowClick: function (r) { selectBiomarker(String(r.symbol)); }
    });
    clear(host);
    host.appendChild(table);
    updateBiomarkerStatus(payload, visible.length);
  }

  function updateBiomarkerStatus(payload, shown) {
    var el = bind('bm-status');
    if (!el) return;
    clear(el);
    var total = payload && numOrNull(payload.total) !== null ? numOrNull(payload.total) : null;
    var returned = payload && numOrNull(payload.returned) !== null ? numOrNull(payload.returned) : null;
    el.appendChild(h('span', { text: 'Showing ' }));
    el.appendChild(h('span', { class: 'num', text: String(shown === undefined ? (returned === null ? 0 : returned) : shown) }));
    el.appendChild(h('span', { text: ' of ' }));
    el.appendChild(h('span', { class: 'num', text: total === null ? (returned === null ? '—' : String(returned)) : String(total) }));
    el.appendChild(h('span', { text: ' ranked features  ·  filter ' + blockLabel(state.bm.block).toLowerCase() }));
    if (state.bm.query) el.appendChild(h('span', { text: '  ·  search "' + state.bm.query.trim() + '"' }));
    if (payload && payload.method) el.appendChild(h('span', { text: '  ·  ' + payload.method }));
  }

  function selectBiomarker(symbol) {
    if (!symbol) return;
    state.selectedSymbol = symbol;
    var host = panel('biomarker-detail');
    if (!host) return;
    setText('bm-drawer-title', symbol);
    var closeBtn = byId('bm-drawer-close');
    if (closeBtn) closeBtn.hidden = false;
    setBusy(host, 'Loading ' + symbol);
    renderBiomarkers(state.biomarkers);

    api.biomarker(symbol).then(function (d) {
      renderBiomarkerDetail(d);
    }, function (err) {
      setError(host, err, 'Feature ' + symbol);
    });
  }

  function renderBiomarkerDetail(d) {
    var host = panel('biomarker-detail');
    if (!host) return;

    if (!d || d.present === false) {
      setEmpty(host, '“' + (d && d.symbol ? d.symbol : '—') + '” is not in the ranked panel. ' +
        'The server returned present=false, which is a miss rather than an error.');
      setText('bm-drawer-title', (d && d.symbol) || 'Feature detail');
      return;
    }

    var title = d.symbol || '—';
    setText('bm-drawer-title', title);

    var rows = [
      specRow('Rank', fmtInt(d.rank)),
      specRow('Block', blockLabel(d.block)),
      specRow('Axis', d.axis),
      specRow('Importance', fmt(d.importance, 4)),
      specRow('Stability', fmt(d.stability, 3)),
      specRow('Direction', directionLabel(d.direction)),
      specRow('Effect size', fmtSigned(d.effect_size, 2)),
      specRow('Permutation p', fmtP(d.p_value)),
      specRow('FDR q (BH)', fmtP(d.q_value)),
      specRow('Multiple testing', d.fdr_significant === true
        ? 'Significant at q < 0.05'
        : 'Not significant at q < 0.05'),
      specRow('Stage I mean', fmtSigned(d.mean_early, 2)),
      specRow('Stage II–IV mean', fmtSigned(d.mean_advanced, 2)),
      specRow('Observed in', numOrNull(d.n_observed) !== null ? fmtInt(d.n_observed) + ' samples' : null)
    ];

    var extra = [];
    [['Mutation frequency, stage I', d.mutation_frequency_early],
     ['Mutation frequency, stage II–IV', d.mutation_frequency_advanced],
     ['Train frequency, stage I', d.train_frequency_early],
     ['Train frequency, stage II–IV', d.train_frequency_advanced]
    ].forEach(function (pair) {
      if (numOrNull(pair[1]) !== null) extra.push(specRow(pair[0], fmt(pair[1], 3)));
    });

    var kind = numOrNull(d.mutation_frequency_early) !== null
      ? 'Mutation and train-set frequencies are reported alongside the expression means. The two ' +
        'are different measurements of the same gene and are not expected to agree.'
      : 'Only standardised feature means are available for this symbol; the server returned no ' +
        'mutation or train-set frequency fields.';

    clear(host);
    host.appendChild(h('p', { class: 'drawer-symbol', text: title }));
    host.appendChild(h('div', { class: 'drawer-grid' }, h('dl', { class: 'spec-list' }, rows.concat(extra))));
    host.appendChild(h('p', { class: 'geo-note', text: kind }));
    host.appendChild(h('div', { class: 'drawer-actions no-print' },
      h('button', {
        type: 'button', class: 'btn btn-quiet btn-small',
        onclick: function () {
          var url = API_BASE + '/api/biomarkers/' + encodeURIComponent(title);
          global.open(url, '_blank', 'noopener');
        }
      }, 'Open raw record')));
  }

  /* ============================================================= /api/cohort */

  function renderCohort(c) {
    state.cohort = c;
    var host = panel('cohort-table');
    if (!host) return;

    var rows = Array.isArray(c.rows) ? c.rows : [];
    renderStageChart(c);

    /* filter option sources, built once from the served distributions */
    syncSelect(byId('co-stage'), 'co-stage-ready', (c.stage_distribution || []).map(function (d) {
      return { value: String(d.stage), label: 'Stage ' + d.stage + '  (' + fmtInt(d.count) + ')' };
    }), 'All stages');

    syncSelect(byId('co-histology'), 'co-histology-ready', (c.histology_distribution || []).map(function (d) {
      return { value: String(d.name), label: String(d.name) + '  (' + fmtInt(d.count) + ')' };
    }), 'All types');

    if (!rows.length) {
      setEmpty(host, 'No sample rows returned for offset ' + state.co.offset + '.');
      renderPager();
      return;
    }

    var q = state.co.query.trim().toLowerCase();
    var visible = rows.filter(function (r) {
      if (state.co.stage && String(r.figo_stage) !== state.co.stage) return false;
      if (state.co.histology && String(r.histological_type) !== state.co.histology) return false;
      if (q && String(r.sample_id || '').toLowerCase().indexOf(q) === -1) return false;
      return true;
    });

    if (!visible.length) {
      setEmpty(host, rows.length + ' rows loaded, none match the current filter.');
      renderPager();
      return;
    }

    var total = numOrNull(c.total);
    var maxConf = visible.reduce(function (acc, r) {
      var v = numOrNull(r.confidence);
      return v === null ? acc : Math.max(acc, v);
    }, 0);

    var table = dataTable({
      columns: [
        { key: 'sample_id', label: 'Sample', align: 'left', sortable: true, render: function (r) { return String(r.sample_id || '—'); } },
        { key: 'figo_stage', label: 'Stage', sortable: true, render: function (r) {
            var cls = stageClass(r.figo_stage);
            return h('span', { class: 'chip ' + (cls === 'early' ? 'chip-early' : cls === 'advanced' ? 'chip-advanced' : 'chip-muted'), text: String(r.figo_label || r.figo_stage || '—') });
          } },
        { key: 'histological_type', label: 'Histology', align: 'left', nowrap: false, sortable: true, render: function (r) { return String(r.histological_type || '—'); } },
        { key: 'age_at_diagnosis', label: 'Age', sortable: true, render: function (r) {
            var v = numOrNull(r.age_at_diagnosis);
            return v === null ? '—' : String(Math.round(v));
          } },
        { key: 'sample_type', label: 'Type', align: 'left', sortable: true, render: function (r) { return titleCase(r.sample_type); } },
        { key: 'prediction', label: 'Prediction', align: 'left', sortable: true, render: function (r) {
            var early = r.prediction === 'early_stage';
            return h('span', { class: 'chip ' + (early ? 'chip-early' : 'chip-advanced'), text: early ? 'stage I' : 'stage II–IV' });
          } },
        { key: 'confidence', label: 'Confidence', sortable: true, render: function (r) {
            return bar(r.confidence, maxConf > 0 ? maxConf : 1, 'neutral');
          } },
        { key: 'in_sample', label: 'In split', sortable: true, render: function (r) {
            return h('span', { class: 'chip ' + (r.in_sample === true ? 'chip-accent' : 'chip-muted'), text: r.in_sample === true ? 'test' : 'held out' });
          } }
      ],
      rows: visible,
      sort: state.co.sort,
      onSort: function (key, dir) { state.co.sort = { key: key, dir: dir }; renderCohort(state.cohort); },
      caption: 'Columns served: ' + (Array.isArray(c.columns) && c.columns.length ? c.columns.join(', ') : '—') +
        '. Sorted and filtered client-side within the loaded page.'
    });
    clear(host);
    host.appendChild(table);
    renderPager();
    renderCohortSummary(c, visible.length);
  }

  function renderCohortSummary(c, shown) {
    var host = panel('cohort-summary');
    if (!host) return;
    clear(host);
    host.appendChild(h('dl', { class: 'spec-list' }, [
      specRow('Cohort total', fmtThousands(c.total)),
      specRow('Rows loaded', fmtInt(state.co.limit)),
      specRow('Rows after filter', fmtInt(shown)),
      specRow('Offset', fmtInt(state.co.offset)),
      specRow('Histology types', fmtInt((c.histology_distribution || []).length))
    ]));
  }

  /**
   * Stage distribution is served with every cohort page, so redraw only when the
   * payload actually changes — otherwise each filter change would attach another
   * ResizeObserver to the same host.
   */
  function renderStageChart(c) {
    var host = byId('chart-stage');
    if (!host) return;
    var dist = Array.isArray(c.stage_distribution) ? c.stage_distribution : [];
    var key = dist.map(function (d) { return String(d.stage) + ':' + String(d.count); }).join('|');
    if (key === state.stageChartKey) return;
    state.stageChartKey = key;

    var items = dist.filter(function (d) { return numOrNull(d.count) !== null; }).map(function (d) {
      var cls = stageClass(d.stage);
      return {
        label: 'Stage ' + d.stage,
        value: numOrNull(d.count),
        valueText: fmtInt(d.count),
        color: cls === 'early' ? QCharts.SERIES[1] : (cls === 'advanced' ? QCharts.SERIES[2] : QCharts.SERIES[4])
      };
    });
    QCharts.hbars(host, items, {
      axisLabel: 'samples in cohort',
      emptyMessage: 'No stage distribution returned by /api/cohort.'
    });
  }

  function renderPager() {
    var host = bind('co-pager');
    if (!host) return;
    var total = state.cohort ? numOrNull(state.cohort.total) : null;
    var offset = state.co.offset;
    var limit = state.co.limit;
    var from = total === null ? offset + 1 : Math.min(offset + 1, total);
    var to = total === null ? offset + limit : Math.min(offset + limit, total);

    clear(host);
    host.appendChild(h('span', {}, [
      h('span', { text: 'Rows ' }), h('span', { class: 'pager-note', text: from + '–' + to }),
      total === null ? null : h('span', { class: 'pager-note', text: ' of ' + fmtThousands(total) })
    ]));
    host.appendChild(h('span', { class: 'pager-actions' }, [
      h('button', {
        type: 'button', class: 'btn btn-quiet btn-small', text: 'Previous',
        disabled: offset <= 0,
        onclick: function () { state.co.offset = Math.max(0, offset - limit); loadCohort(); }
      }),
      h('button', {
        type: 'button', class: 'btn btn-quiet btn-small', text: 'Next',
        disabled: total === null ? false : offset + limit >= total,
        onclick: function () { state.co.offset = offset + limit; loadCohort(); }
      })
    ]));
  }

  function syncSelect(sel, flag, options, allLabel) {
    if (!sel) return;
    if (sel.dataset[flag]) {
      /* keep current value across re-renders */
      var keep = sel.value;
      if (options.some(function (o) { return o.value === keep; })) return;
    }
    sel.dataset[flag] = '1';
    clear(sel);
    sel.appendChild(h('option', { value: '', text: allLabel }));
    options.forEach(function (o) { sel.appendChild(h('option', { value: o.value, text: o.label })); });
  }

  /* ============================================================= /api/predict */

  var MODALITIES = ['transcriptome', 'genome', 'proteome'];

  function buildPredictor() {
    MODALITIES.forEach(function (mod) {
      var rows = doc.querySelector('[data-rows="' + mod + '"]');
      if (rows) { clear(rows); rows.appendChild(featureRow(mod)); }
    });
    MODALITIES.forEach(function (mod) { updateFeatureCount(mod); });

    Array.prototype.forEach.call(doc.querySelectorAll('[data-add]'), function (btn) {
      btn.addEventListener('click', function () {
        var mod = btn.getAttribute('data-add');
        var host = doc.querySelector('[data-rows="' + mod + '"]');
        if (host) { host.appendChild(featureRow(mod)); updateFeatureCount(mod); }
      });
    });

    var json = byId('predict-json');
    if (json) {
      json.value = JSON.stringify({
        transcriptome: { PTEN: -1.2, MKI67: 0.4 },
        genome: { 'MUT::PTEN': 1, 'CNA::TP53': -1 },
        proteome: { 'pAKT_pS473-R-V': 0.8 }
      }, null, 2);
    }

    Array.prototype.forEach.call(doc.querySelectorAll('[data-mode]'), function (btn) {
      btn.addEventListener('click', function () { setPredictMode(btn.getAttribute('data-mode')); });
    });

    var form = byId('predict-form');
    if (form) form.addEventListener('submit', onPredictSubmit);

    var reset = byId('predict-reset');
    if (reset) {
      reset.addEventListener('click', function () {
        var host = panel('predict-result');
        if (host) {
          clear(host);
          host.appendChild(h('p', { class: 'muted-note', text: 'No prediction requested.' }));
        }
        setText('predict-meta', null);
        state.lastPrediction = null;
      });
    }
  }

  function featureRow(mod) {
    var nameId = 'f-' + mod + '-name-' + Math.random().toString(36).slice(2, 8);
    var valId = 'f-' + mod + '-val-' + Math.random().toString(36).slice(2, 8);
    var name = h('input', {
      class: 'input', type: 'text', id: nameId, placeholder: 'feature name',
      autocomplete: 'off', spellcheck: 'false', 'aria-label': mod + ' feature name'
    });
    var value = h('input', {
      class: 'input', type: 'number', id: valId, step: 'any', placeholder: 'value',
      'aria-label': mod + ' feature value'
    });
    var remove = h('button', {
      type: 'button', class: 'feature-remove', text: '×',
      'aria-label': 'Remove this ' + mod + ' feature row'
    });
    remove.addEventListener('click', function () {
      var host = doc.querySelector('[data-rows="' + mod + '"]');
      if (host) {
        if (host.children.length > 1) remove.parentNode.remove();
        else { name.value = ''; value.value = ''; }
        updateFeatureCount(mod);
      }
    });
    [name, value].forEach(function (input) {
      input.addEventListener('input', function () { updateFeatureCount(mod); });
    });
    return h('div', { class: 'feature-row' }, [name, value, remove]);
  }

  function updateFeatureCount(mod) {
    var el = doc.querySelector('[data-count="' + mod + '"]');
    var host = doc.querySelector('[data-rows="' + mod + '"]');
    if (!el || !host) return;
    var filled = 0;
    Array.prototype.forEach.call(host.querySelectorAll('.feature-row'), function (row) {
      var inputs = row.querySelectorAll('input');
      if (inputs.length >= 2 && inputs[0].value.trim() !== '' && inputs[1].value.trim() !== '') filled++;
    });
    el.textContent = String(filled);
  }

  function setPredictMode(mode) {
    state.predictMode = mode === 'raw' ? 'raw' : 'structured';
    Array.prototype.forEach.call(doc.querySelectorAll('[data-mode]'), function (btn) {
      btn.setAttribute('aria-pressed', btn.getAttribute('data-mode') === state.predictMode ? 'true' : 'false');
    });
    var s = byId('predict-structured');
    var r = byId('predict-raw');
    if (s) s.hidden = state.predictMode !== 'structured';
    if (r) r.hidden = state.predictMode !== 'raw';
  }

  function collectSample() {
    var sample = { transcriptome: {}, genome: {}, proteome: {} };
    var problems = [];
    MODALITIES.forEach(function (mod) {
      var host = doc.querySelector('[data-rows="' + mod + '"]');
      if (!host) return;
      Array.prototype.forEach.call(host.querySelectorAll('.feature-row'), function (row) {
        var inputs = row.querySelectorAll('input');
        if (inputs.length < 2) return;
        var name = inputs[0].value.trim();
        var raw = inputs[1].value.trim();
        if (name === '' && raw === '') return;
        if (name === '') { problems.push(mod + ': a value was supplied without a feature name'); return; }
        if (raw === '') { problems.push(mod + ': ' + name + ' has no value'); return; }
        var n = parseFloat(raw);
        if (!isFinite(n)) { problems.push(mod + ': ' + name + ' value "' + raw + '" is not a number'); return; }
        sample[mod][name] = n;
      });
    });
    return { sample: sample, problems: problems };
  }

  function onPredictSubmit(ev) {
    ev.preventDefault();
    var host = panel('predict-result');
    var submit = byId('predict-submit');
    if (!host) return;

    var body;
    if (state.predictMode === 'raw') {
      var jsonEl = byId('predict-json');
      var text = jsonEl ? jsonEl.value.trim() : '';
      if (!text) { setPredictError(host, 'The raw JSON box is empty.'); return; }
      var parsed;
      try { parsed = JSON.parse(text); }
      catch (e) { setPredictError(host, 'The raw JSON box is not valid JSON: ' + e.message); return; }
      if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
        setPredictError(host, 'The sample object must be a JSON object keyed by modality.');
        return;
      }
      body = { sample: parsed };
    } else {
      var collected = collectSample();
      if (collected.problems.length) {
        setPredictError(host, 'Fix the structured input: ' + collected.problems.slice(0, 4).join('; '));
        return;
      }
      var count = MODALITIES.reduce(function (acc, m) { return acc + Object.keys(collected.sample[m]).length; }, 0);
      if (count === 0) {
        setPredictError(host, 'Supply at least one feature value, or switch to the raw JSON input.');
        return;
      }
      body = { sample: collected.sample };
    }

    var topK = numOrNull(byId('predict-topk') ? byId('predict-topk').value : null);
    body.top_k = topK === null ? 10 : Math.max(1, Math.min(50, Math.round(topK)));

    if (submit) submit.disabled = true;
    setBusy(host, 'Running inference');

    api.predict(body).then(function (r) {
      state.lastPrediction = r;
      if (submit) submit.disabled = false;
      renderPrediction(r);
    }, function (err) {
      if (submit) submit.disabled = false;
      setError(host, err, 'Prediction');
    });
  }

  function setPredictError(host, message) {
    clear(host);
    host.appendChild(h('div', { class: 'state state--error', role: 'alert' }, [
      h('p', { class: 'state-title', text: 'Input rejected' }),
      h('p', { text: message })
    ]));
  }

  function renderPrediction(r) {
    var host = panel('predict-result');
    if (!host || !r) return;

    var probs = r.probabilities || {};
    var pEarly = numOrNull(probs.stage_I_early);
    var pAdv = numOrNull(probs.stage_II_IV_advanced);
    if (pEarly === null && pAdv !== null) pEarly = Math.max(0, 1 - pAdv);
    if (pAdv === null && pEarly !== null) pAdv = Math.max(0, 1 - pEarly);

    var early = r.prediction === 'early_stage';
    var contributors = Array.isArray(r.top_contributors) ? r.top_contributors : [];

    setText('predict-meta', numOrNull(r.matched_features) !== null
      ? numOrNull(r.matched_features) + ' of ' + fmtInt(r.total_features) + ' features matched'
      : null);

    var maxContrib = contributors.reduce(function (acc, c) {
      var v = numOrNull(c.contribution);
      return v === null ? acc : Math.max(acc, Math.abs(v));
    }, 0);

    clear(host);

    host.appendChild(h('div', { class: 'result-head' }, [
      h('div', null, [
        h('p', { class: 'result-label', text: String(r.label || r.prediction || '—') }),
        h('p', { class: 'result-sub', text: 'confidence ' + fmt(r.confidence, 4) })
      ]),
      h('span', { class: 'result-tier', 'data-tier': String(r.risk_tier || ''), text: String(r.risk_tier || 'tier n/a') })
    ]));

    host.appendChild(h('div', { class: 'prob-bar', role: 'img', 'aria-label':
      'Probability ' + fmtP(pEarly) + ' stage I, ' + fmtP(pAdv) + ' stage II to IV' }, [
      h('span', { class: 'prob-seg early', style: 'width:' + (QCharts.isNum(pEarly) ? (pEarly * 100).toFixed(2) : '50') + '%' }),
      h('span', { class: 'prob-seg advanced', style: 'width:' + (QCharts.isNum(pAdv) ? (pAdv * 100).toFixed(2) : '50') + '%' })
    ]));
    host.appendChild(h('div', { class: 'prob-legend' }, [
      h('span', { text: 'stage I ' + fmtP(pEarly) }),
      h('span', { text: 'stage II–IV ' + fmtP(pAdv) })
    ]));

    host.appendChild(h('div', { class: 'drawer-grid' }, h('dl', { class: 'spec-list' }, [
      specRow('Coverage', fmtPct(r.coverage, 3)),
      specRow('Matched features', fmtInt(r.matched_features)),
      specRow('Model features', fmtThousands(r.total_features)),
      specRow('Imputation', 'training median for every unset feature')
    ])));

    if (contributors.length) {
      var table = dataTable({
        columns: [
          { key: 'name', label: 'Feature', align: 'left', render: function (c) { return String(c.name || '—'); } },
          { key: 'block', label: 'Block', align: 'left', render: function (c) { return h('span', { class: 'chip', text: blockLabel(c.block) }); } },
          { key: 'value', label: 'Value', sortable: true, render: function (c) { return fmt(c.value, 3); } },
          { key: 'importance', label: 'Importance', sortable: true, render: function (c) { return fmt(c.importance, 4); } },
          { key: 'contribution', label: 'Contribution', sortable: true, render: function (c) {
              return bar(c.contribution, maxContrib > 0 ? maxContrib : 1, early ? 'early' : 'advanced');
            } },
          { key: 'direction', label: 'Direction', align: 'left', render: function (c) { return directionLabel(c.direction); } }
        ],
        rows: contributors,
        sort: { key: 'contribution', dir: 'desc' },
        caption: 'Top contributors by absolute contribution to the returned probability.'
      });
      host.appendChild(h('div', { class: 'drawer-grid' }, table));
    } else {
      host.appendChild(h('p', { class: 'geo-note', text: 'The server returned no top contributors for this sample.' }));
    }

    host.appendChild(h('p', { class: 'disclaimer', text: String(r.disclaimer || 'Research use only. Not a diagnostic device.') }));
  }

  /* --------------------------------------------------------------- loaders */

  function loadPanel(name, key, fn) {
    var host = panel(name);
    setLoading(host, 5);
    return api[key]().then(function (data) {
      fn(data);
      return data;
    }, function (err) {
      setError(host, err, name);
      return null;
    });
  }

  function loadBiomarkers() {
    var host = panel('biomarkers-table');
    setLoading(host, 8);
    return api.biomarkers({ limit: state.bm.limit, block: state.bm.block }).then(function (data) {
      renderBiomarkers(data);
    }, function (err) {
      setError(host, err, 'Biomarkers');
      updateBiomarkerStatus(null);
    });
  }

  function loadCohort() {
    var host = panel('cohort-table');
    setLoading(host, 8);
    return api.cohort({ limit: state.co.limit, offset: state.co.offset }).then(function (data) {
      renderCohort(data);
    }, function (err) {
      setError(host, err, 'Cohort');
      var pager = bind('co-pager');
      if (pager) clear(pager);
    });
  }

  function downloadCohortCsv(btn) {
    if (!btn) return;
    var original = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Preparing…';
    api.cohortDownload().then(function (blob) {
      var href = URL.createObjectURL(blob);
      var a = h('a', { href: href, download: 'qomics_tcga_ucec_cohort.csv' });
      doc.body.appendChild(a);
      a.click();
      doc.body.removeChild(a);
      global.setTimeout(function () { URL.revokeObjectURL(href); }, 2000);
      btn.textContent = 'Downloaded';
      global.setTimeout(function () { btn.textContent = original; }, 2200);
    }, function (err) {
      var d = describeError(err);
      var host = panel('cohort-summary');
      if (host) {
        host.appendChild(h('p', { class: 'state-code', text: 'CSV download failed — ' + d.title + ' ' + (d.code || '') }));
      }
      btn.textContent = original;
    }).then(function () { btn.disabled = false; });
  }

  /* ------------------------------------------------------------ scroll-spy */

  function initScrollSpy() {
    var links = {};
    Array.prototype.forEach.call(doc.querySelectorAll('[data-spy]'), function (a) {
      links[a.getAttribute('data-spy')] = a;
    });

    /* section id -> nav key it should highlight */
    var map = {
      overview: 'overview',
      method: 'method',
      quantum: 'method',
      results: 'results',
      biomarkers: 'biomarkers',
      cohort: 'cohort',
      predict: 'cohort',
      architecture: 'architecture'
    };

    var pending = false;
    function update() {
      pending = false;
      var headerH = (byId('section-nav') && byId('section-nav').closest('.masthead'))
        ? byId('section-nav').closest('.masthead').offsetHeight : 60;
      var probe = global.scrollY + headerH + 48;
      var best = null;
      Object.keys(map).forEach(function (id) {
        var sec = byId(id);
        if (!sec) return;
        if (sec.offsetTop <= probe) best = id;
      });
      var key = best ? map[best] : null;
      Object.keys(links).forEach(function (k) {
        if (k === key) links[k].setAttribute('aria-current', 'true');
        else links[k].removeAttribute('aria-current');
      });
    }

    function onScroll() {
      if (pending) return;
      pending = true;
      (global.requestAnimationFrame || setTimeout)(update);
    }

    global.addEventListener('scroll', onScroll, { passive: true });
    global.addEventListener('resize', onScroll);
    update();
  }

  /* ------------------------------------------------------------------ wire */

  function initToolbar() {
    Array.prototype.forEach.call(doc.querySelectorAll('[data-block]'), function (btn) {
      btn.addEventListener('click', function () {
        state.bm.block = btn.getAttribute('data-block');
        Array.prototype.forEach.call(doc.querySelectorAll('[data-block]'), function (b) {
          b.setAttribute('aria-pressed', b.getAttribute('data-block') === state.bm.block ? 'true' : 'false');
        });
        loadBiomarkers();
      });
    });

    var search = byId('bm-search');
    if (search) {
      var timer = null;
      search.addEventListener('input', function () {
        state.bm.query = search.value;
        if (timer) global.clearTimeout(timer);
        timer = global.setTimeout(function () { renderBiomarkers(state.biomarkers); }, 120);
      });
    }

    var limit = byId('bm-limit');
    if (limit) {
      limit.addEventListener('change', function () {
        state.bm.limit = numOrNull(limit.value) || 50;
        loadBiomarkers();
      });
    }

    var reset = byId('bm-reset');
    if (reset) {
      reset.addEventListener('click', function () {
        state.bm = { block: 'all', limit: numOrNull(limit && limit.value) || 50, query: '', sort: { key: 'rank', dir: 'asc' } };
        if (search) search.value = '';
        Array.prototype.forEach.call(doc.querySelectorAll('[data-block]'), function (b) {
          b.setAttribute('aria-pressed', b.getAttribute('data-block') === 'all' ? 'true' : 'false');
        });
        closeDrawer();
        loadBiomarkers();
      });
    }

    var print = byId('bm-print');
    if (print) print.addEventListener('click', function () { global.print(); });

    var close = byId('bm-drawer-close');
    if (close) close.addEventListener('click', closeDrawer);

    var stage = byId('co-stage');
    if (stage) stage.addEventListener('change', function () { state.co.stage = stage.value; renderCohort(state.cohort); });

    var hist = byId('co-histology');
    if (hist) hist.addEventListener('change', function () { state.co.histology = hist.value; renderCohort(state.cohort); });

    var coSearch = byId('co-search');
    if (coSearch) {
      var t2 = null;
      coSearch.addEventListener('input', function () {
        state.co.query = coSearch.value;
        if (t2) global.clearTimeout(t2);
        t2 = global.setTimeout(function () { renderCohort(state.cohort); }, 120);
      });
    }

    var coLimit = byId('co-limit');
    if (coLimit) {
      coLimit.addEventListener('change', function () {
        state.co.limit = numOrNull(coLimit.value) || 60;
        state.co.offset = 0;
        loadCohort();
      });
    }

    var dl = byId('co-download');
    if (dl) dl.addEventListener('click', function () { downloadCohortCsv(dl); });
  }

  function closeDrawer() {
    state.selectedSymbol = null;
    var host = panel('biomarker-detail');
    if (host) {
      clear(host);
      host.appendChild(h('p', { class: 'muted-note', text: 'Select a row to load its provenance record.' }));
    }
    setText('bm-drawer-title', 'Feature detail');
    var close = byId('bm-drawer-close');
    if (close) close.hidden = true;
    if (state.biomarkers) renderBiomarkers(state.biomarkers);
  }

  /* ------------------------------------------------------------------ boot */

  function boot() {
    initToolbar();
    buildPredictor();
    setPredictMode('structured');
    initScrollSpy();

    api.health().then(function (health) {
      hideGate();
      renderHealthStatus(health);
      if (health && health.model_loaded !== true) {
        notice({
          title: 'Server up, model artefacts not loaded',
          body: 'Panels that depend on the trained checkpoint will report 503 until scripts/train.py has been run. ' +
            'Static panels remain available.'
        });
      }

      return Promise.all([
        loadPanel('overview-cohort', 'overview', renderOverview),
        loadPanel('metrics-table', 'metrics', renderMetrics),
        loadPanel('pipeline', 'architecture', renderArchitecture),
        loadPanel('features', 'features', renderFeatures),
        loadPanel('quantum-register', 'quantum', renderQuantum),
        loadBiomarkers(),
        loadCohort()
      ]);
    }, function (err) {
      var dot = bind('status-dot');
      var missing = err && err.status === 503;
      if (missing) {
        showGate('HTTP 503 from /api/health', 'Model artefacts not present');
        if (dot) dot.setAttribute('data-state', 'degraded');
        setText('status-text', 'degraded · artefacts missing');
      } else {
        showGate((err && err.message) || 'unknown error');
        if (dot) dot.setAttribute('data-state', 'down');
        setText('status-text', 'offline');
      }
      document.querySelectorAll('[data-panel]').forEach(function (el) {
        setError(el, err, el.getAttribute('data-panel'));
      });
    });
  }

  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', boot);
  else boot();

  /* exported for the mock harness in _scratch (not used by the page) */
  global.QOMICS = {
    api: api,
    state: state,
    renderOverview: renderOverview,
    renderMetrics: renderMetrics,
    renderArchitecture: renderArchitecture,
    renderFeatures: renderFeatures,
    renderQuantum: renderQuantum,
    renderBiomarkers: renderBiomarkers,
    renderCohort: renderCohort,
    renderPrediction: renderPrediction,
    dataTable: dataTable,
    fmt: fmt,
    fmtInt: fmtInt,
    fmtPct: fmtPct,
    fmtSigned: fmtSigned
  };
}(typeof window !== 'undefined' ? window : globalThis));

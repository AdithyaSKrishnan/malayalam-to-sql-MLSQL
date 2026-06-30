document.addEventListener('DOMContentLoaded', () => {

    // ─── Element References ────────────────────────────
    const tabs = document.querySelectorAll('.tab');
    const panels = document.querySelectorAll('.tab-panel');
    const tabIndicator = document.getElementById('tabIndicator');

    // SQL tab elements
    const sqlInput = document.getElementById('sqlInput');
    const sqlSchemaInput = document.getElementById('sqlSchemaInput');
    const convertBtn = document.getElementById('convertBtn');
    const sqlMongoOutput = document.getElementById('sqlMongoOutput');
    const sqlCopyBtn = document.getElementById('sqlCopyBtn');

    // Malayalam tab elements
    const malayalamInput = document.getElementById('malayalamInput');
    const mlSchemaInput = document.getElementById('mlSchemaInput');
    const malayalamConvertBtn = document.getElementById('malayalamConvertBtn');
    const mlSqlOutput = document.getElementById('mlSqlOutput');
    const mlMongoOutput = document.getElementById('mlMongoOutput');
    const mlSqlCopyBtn = document.getElementById('mlSqlCopyBtn');
    const mlMongoCopyBtn = document.getElementById('mlMongoCopyBtn');

    // ─── Default Schema ────────────────────────────────
    const defaultSchema = JSON.stringify({
        "student": {
            "type": "collection",
            "embeds": ["course"],
            "references": ["department"]
        },
        "department": {
            "type": "collection"
        }
    }, null, 2);

    // Pre-fill schemas
    if (!sqlSchemaInput.value) sqlSchemaInput.value = defaultSchema;
    if (!mlSchemaInput.value) mlSchemaInput.value = defaultSchema;

    // ─── Tab Switching ─────────────────────────────────
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;

            // Update active tab
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Move indicator
            if (target === 'malayalam') {
                tabIndicator.classList.add('at-malayalam');
            } else {
                tabIndicator.classList.remove('at-malayalam');
            }

            // Switch panels
            panels.forEach(p => {
                p.classList.remove('active');
                if (p.dataset.panel === target) {
                    p.classList.add('active');
                }
            });
        });
    });

    // ─── Helper: Toggle Button Loading State ───────────
    function setLoading(button, isLoading) {
        const content = button.querySelector('.btn-content');
        const loader = button.querySelector('.btn-loader');
        if (isLoading) {
            content.hidden = true;
            loader.hidden = false;
            button.disabled = true;
        } else {
            content.hidden = false;
            loader.hidden = true;
            button.disabled = false;
        }
    }

    // ─── Helper: Set Output ────────────────────────────
    function setOutput(element, text, type = 'success') {
        element.textContent = text;
        element.className = '';
        if (type === 'error') {
            element.classList.add('output-error');
        } else if (type === 'placeholder') {
            element.classList.add('output-placeholder');
        } else {
            element.classList.add('output-success');
        }
    }

    // ─── Helper: Copy to Clipboard ─────────────────────
    function setupCopy(btn, getTextFn) {
        btn.addEventListener('click', () => {
            const text = getTextFn();
            if (!text || text.startsWith('//')) return;

            navigator.clipboard.writeText(text).then(() => {
                const copyIcon = btn.querySelector('.copy-icon');
                const checkIcon = btn.querySelector('.check-icon');
                copyIcon.hidden = true;
                checkIcon.hidden = false;
                setTimeout(() => {
                    copyIcon.hidden = false;
                    checkIcon.hidden = true;
                }, 2000);
            });
        });
    }

    // Wire up copy buttons
    setupCopy(sqlCopyBtn, () => sqlMongoOutput.textContent);
    setupCopy(mlSqlCopyBtn, () => mlSqlOutput.textContent);
    setupCopy(mlMongoCopyBtn, () => mlMongoOutput.textContent);

    // ─── SQL → NoSQL Conversion ────────────────────────
    convertBtn.addEventListener('click', async () => {
        const sql = sqlInput.value.trim();
        const schema = sqlSchemaInput.value.trim();

        if (!sql) {
            setOutput(sqlMongoOutput, '⚠ Please enter a SQL query.', 'error');
            return;
        }

        setLoading(convertBtn, true);
        setOutput(sqlMongoOutput, '// Converting…', 'placeholder');

        try {
            const response = await fetch('/api/convert', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sql, schema })
            });

            const data = await response.json();

            if (data.error) {
                setOutput(sqlMongoOutput, '✗ ' + data.error, 'error');
            } else {
                setOutput(sqlMongoOutput, data.mongoQuery, 'success');
            }
        } catch (err) {
            setOutput(sqlMongoOutput, '✗ Network Error: ' + err.message, 'error');
        } finally {
            setLoading(convertBtn, false);
        }
    });

    // ─── Malayalam → NoSQL Full Pipeline ────────────────
    malayalamConvertBtn.addEventListener('click', async () => {
        const question = malayalamInput.value.trim();
        const schema = mlSchemaInput.value.trim();

        if (!question) {
            setOutput(mlSqlOutput, '⚠ Please enter a Malayalam question.', 'error');
            setOutput(mlMongoOutput, '', 'placeholder');
            return;
        }

        setLoading(malayalamConvertBtn, true);
        setOutput(mlSqlOutput, '// Translating Malayalam → SQL…', 'placeholder');
        setOutput(mlMongoOutput, '// Waiting for SQL conversion…', 'placeholder');

        try {
            const response = await fetch('/api/malayalam-to-nosql', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, schema })
            });

            const data = await response.json();

            if (data.error) {
                setOutput(mlSqlOutput, '✗ ' + data.error, 'error');
                setOutput(mlMongoOutput, '', 'placeholder');
            } else {
                // Show intermediate SQL
                if (data.generatedSql) {
                    setOutput(mlSqlOutput, data.generatedSql, 'success');
                } else {
                    setOutput(mlSqlOutput, '(No SQL generated)', 'placeholder');
                }

                // Show final MongoDB query
                if (data.mongoQuery) {
                    setOutput(mlMongoOutput, data.mongoQuery, 'success');
                } else {
                    setOutput(mlMongoOutput, '(No MongoDB query generated)', 'placeholder');
                }
            }
        } catch (err) {
            setOutput(mlSqlOutput, '✗ Network Error: ' + err.message, 'error');
            setOutput(mlMongoOutput, '✗ Could not reach the server. Is the inference API running?', 'error');
        } finally {
            setLoading(malayalamConvertBtn, false);
        }
    });

    // ─── Keyboard shortcut: Ctrl+Enter to convert ──────
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const activePanel = document.querySelector('.tab-panel.active');
            if (activePanel.dataset.panel === 'sql') {
                convertBtn.click();
            } else {
                malayalamConvertBtn.click();
            }
        }
    });

});

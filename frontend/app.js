/**
 * Live Agent Supervision Dashboard
 * Main Application Logic
 */

class AgentSupervisionDashboard {
    constructor() {
        // DOM Elements
        this.canvasContainer = document.getElementById('canvasContainer');
        this.nodesContainer = document.getElementById('nodesContainer');
        this.connectionsLayer = document.getElementById('connectionsLayer');
        this.minimap = document.getElementById('minimap');
        this.minimapViewport = document.getElementById('minimapViewport');

        // State
        this.nodes = new Map();
        this.connections = [];
        this.zoom = 1;
        this.pan = { x: 0, y: 0 };
        this.isDragging = false;
        this.dragNode = null;
        this.dragOffset = { x: 0, y: 0 };

        // SSE state
        this.eventSource = null;
        this.currentRunId = this.getRunIdFromUrl();

        // Agent name to node ID mapping for SSE events
        this.agentNameToNodeId = {
            'ML_EFFORT': 'route_planner',  // Maps to route planner visualization
            'ROUTE_PLANNER': 'route_planner',
            'FAIRNESS_MANAGER': 'fairness_manager',
            'DRIVER_LIAISON': 'driver_liaison',
            'EXPLAINABILITY': 'explainability',
            'FINAL_RESOLUTION': 'route_planner',
        };

        // Layout config
        this.layout = this.getDefaultLayout();

        // Initialize
        this.init();
    }

    /**
     * Get run_id from URL query parameter
     */
    getRunIdFromUrl() {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('run_id');
    }

    /**
     * Get default node positions
     */
    getDefaultLayout() {
        const containerWidth = window.innerWidth;
        const containerHeight = window.innerHeight - 100; // Account for header/footer

        const centerX = containerWidth / 2;
        const topY = 80;
        const bottomY = topY + 220;

        // Calculate spacing for 6 bottom nodes
        const nodeWidth = 180;
        const spacing = 20;
        const totalWidth = 6 * nodeWidth + 5 * spacing;
        const startX = (containerWidth - totalWidth) / 2;

        return {
            central_orchestrator: { x: centerX - 130, y: topY },
            route_database: { x: startX, y: bottomY },
            route_planner: { x: startX + nodeWidth + spacing, y: bottomY },
            fairness_manager: { x: startX + 2 * (nodeWidth + spacing), y: bottomY },
            driver_liaison: { x: startX + 3 * (nodeWidth + spacing), y: bottomY },
            explainability: { x: startX + 4 * (nodeWidth + spacing), y: bottomY },
            learning: { x: startX + 5 * (nodeWidth + spacing), y: bottomY }
        };
    }

    /**
     * Initialize the dashboard
     */
    async init() {
        // Load saved layout or use default
        this.loadLayout();

        // Get initial workflow state (try real data first, fallback to mock)
        // IMPORTANT: If we have a run_id from URL, fetch data for THAT specific run
        let state;
        try {
            if (this.currentRunId) {
                // Fetch workflow state for the specific run from URL
                state = await API.getRealWorkflowState(null, this.currentRunId);
            } else {
                // No specific run - get latest for today
                state = await API.getRealWorkflowState();
            }
        } catch (error) {
            console.log('Using mock data for initial load:', error.message);
            state = API.getMockWorkflowState();
        }

        // Store current state
        this.currentWorkflowState = state;

        // Render agents
        this.renderAgents(state.agents);

        // Draw connections
        this.updateConnections(state.connections);

        // Update stats
        this.updateStats(state.stats);

        // Setup event listeners
        this.setupEventListeners();

        // Setup minimap
        this.updateMinimap();

        // Inject state machine badges (tick/error) into agent nodes
        this.injectStateBadges();

        // Start API polling with workflow updates
        this.setupAPIPolling();

        // Connect to SSE for real-time agent events
        this.connectSSE();

        // Show run_id banner or no-run-selected message
        if (this.currentRunId) {
            this.showRunBanner(this.currentRunId);
            this.fetchRoutesForRun(this.currentRunId);
            this.fetchRunSummary(this.currentRunId);
            // Fetch and replay historical events (for when allocation already completed)
            this.fetchHistoricalEvents(this.currentRunId);
        } else {
            this.showNoRunSelectedBanner();
        }

        // Show data source indicator
        if (state.isRealData) {
            this.showToast('Connected to live backend data!');
        }
    }

    /**
     * Show banner with current run ID
     */
    showRunBanner(runId) {
        const shortId = runId.substring(0, 8);
        const banner = document.createElement('div');
        banner.id = 'run-banner';
        banner.className = 'run-banner';
        banner.innerHTML = `
            <div class="run-banner-content">
                <span class="run-icon">🎯</span>
                <span>Tracking Run: <strong>${shortId}...</strong></span>
                <button class="run-banner-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        banner.style.cssText = `
            position: fixed;
            top: 70px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
            color: #0a0f1a;
            padding: 8px 20px;
            border-radius: 20px;
            z-index: 1000;
            font-size: 14px;
            font-weight: 500;
            box-shadow: 0 4px 20px rgba(0, 212, 170, 0.3);
        `;
        document.body.appendChild(banner);
    }

    /**
     * Show banner indicating no run is selected
     */
    showNoRunSelectedBanner() {
        const banner = document.createElement('div');
        banner.id = 'no-run-banner';
        banner.innerHTML = `
            <div class="no-run-content">
                <span class="no-run-icon">⚠️</span>
                <span>No allocation run selected. Open from <a href="http://localhost:8000/demo/allocate" target="_blank">API Demo</a> or enter a Run ID.</span>
            </div>
        `;
        banner.style.cssText = `
            position: fixed;
            top: 70px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(251, 191, 36, 0.2);
            color: #fbbf24;
            padding: 12px 24px;
            border-radius: 8px;
            border: 1px solid #fbbf24;
            z-index: 1000;
            font-size: 14px;
        `;
        const link = banner.querySelector('a');
        if (link) {
            link.style.color = '#00d4aa';
            link.style.textDecoration = 'underline';
        }
        document.body.appendChild(banner);
    }

    /**
     * Fetch routes for map from run-scoped endpoint
     */
    async fetchRoutesForRun(runId) {
        try {
            const response = await fetch(`http://localhost:8000/api/v1/runs/${runId}/routes-on-map`);
            if (!response.ok) {
                console.error('Failed to fetch routes for run:', response.status);
                return;
            }
            const data = await response.json();
            console.log('Fetched routes for run:', data.routes.length);

            // Store routes for map display
            this.currentRoutes = data.routes;

            // Update map if available
            if (typeof this.updateMapWithRoutes === 'function') {
                this.updateMapWithRoutes(data.routes);
            }
        } catch (error) {
            console.error('Error fetching routes for run:', error);
        }
    }

    /**
     * Fetch run summary to update stats
     */
    async fetchRunSummary(runId) {
        try {
            const response = await fetch(`http://localhost:8000/api/v1/runs/${runId}/summary`);
            if (!response.ok) {
                console.error('Failed to fetch run summary:', response.status);
                return;
            }
            const data = await response.json();
            console.log('Fetched run summary:', data);

            // Update stats display with run metrics
            this.updateStats({
                total_drivers: data.num_drivers,
                active_routes: data.num_routes,
                delivered: 0,  // Not tracked in summary
                pending: data.num_packages,
                gini_index: data.global_gini_index,
                std_dev: data.global_std_dev,
            });
        } catch (error) {
            console.error('Error fetching run summary:', error);
        }
    }

    /**
     * Fetch and replay historical events for a run
     * This updates agent statuses when the dashboard is opened after allocation completes
     */
    async fetchHistoricalEvents(runId) {
        try {
            const response = await fetch(`http://localhost:8000/api/v1/runs/${runId}/recent-events`);
            if (!response.ok) {
                console.log('No recent events endpoint or no events:', response.status);
                return;
            }
            const data = await response.json();
            const events = data.events || [];

            console.log(`Replaying ${events.length} historical events for run`);

            // Process each event to update agent statuses
            // Sort by timestamp to replay in order
            events.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

            // Track the last state for each agent to show final status
            const agentStates = new Map();
            events.forEach(event => {
                const key = event.agent_name;
                agentStates.set(key, event);
            });

            // Update each agent to its final state
            agentStates.forEach((event, agentName) => {
                this.handleAgentEvent(event);
            });

            if (events.length > 0) {
                this.showToast(`Loaded ${events.length} agent events for this run`);
            }
        } catch (error) {
            console.log('Could not fetch historical events:', error.message);
        }
    }

    /**
     * Update map with routes from run-scoped API
     * Uses driver_name directly from the API response
     */
    updateMapWithRoutes(routes) {
        if (!this.map || !this.mapLayer) {
            console.log('Map not initialized, skipping route update');
            return;
        }

        // Clear existing layers
        this.mapLayer.clearLayers();

        // Route colors palette
        const routeColors = [
            { main: '#00d4aa', glow: '#00b894' },
            { main: '#6c5ce7', glow: '#a29bfe' },
            { main: '#fd79a8', glow: '#e84393' },
            { main: '#fdcb6e', glow: '#f39c12' },
            { main: '#74b9ff', glow: '#0984e3' },
            { main: '#ff7675', glow: '#d63031' },
            { main: '#55efc4', glow: '#00cec9' },
            { main: '#fab1a0', glow: '#e17055' },
        ];

        // Add warehouse marker
        const warehouseIcon = L.divIcon({
            className: 'warehouse-marker',
            html: `
                <div style="
                    width:20px; height:20px;
                    background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
                    border-radius:50%;
                    border:2px solid #fff;
                    box-shadow: 0 0 20px rgba(0,212,170,0.6), 0 4px 8px rgba(0,0,0,0.3);
                    display:flex; align-items:center; justify-content:center;
                ">
                    <span style="color:#fff; font-size:10px; font-weight:bold;">W</span>
                </div>
            `,
            iconSize: [20, 20],
            iconAnchor: [10, 10]
        });

        L.marker([12.9716, 77.5946], { icon: warehouseIcon })
            .addTo(this.mapLayer)
            .bindPopup(`
                <div style="font-family: 'Inter', sans-serif; padding: 8px;">
                    <div style="font-size:14px; font-weight:600; color:#00d4aa; margin-bottom:4px;">
                        🏭 Central Warehouse
                    </div>
                    <div style="font-size:12px; color:#888;">Bangalore Hub</div>
                </div>
            `);

        const renderedRoutes = [];

        // Render each route
        routes.forEach((route, i) => {
            const colorScheme = routeColors[i % routeColors.length];
            // Use driver_name from API response directly
            const driverName = route.driver_name || `Driver ${i + 1}`;

            if (route.stops && route.stops.length > 0) {
                const latlngs = [[12.9716, 77.5946]]; // Start at warehouse

                // Add stop markers
                route.stops.forEach((stop, idx) => {
                    const latlng = [stop.lat, stop.lng];
                    latlngs.push(latlng);

                    const stopIcon = L.divIcon({
                        className: 'stop-marker-premium',
                        html: `
                            <div style="
                                width:24px; height:24px;
                                background: linear-gradient(135deg, ${colorScheme.main} 0%, ${colorScheme.glow} 100%);
                                border-radius:50%;
                                border:2px solid #fff;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.3), 0 0 12px ${colorScheme.main}40;
                                display:flex; align-items:center; justify-content:center;
                                font-size:10px; font-weight:bold; color:#fff;
                            ">${idx + 1}</div>
                        `,
                        iconSize: [24, 24],
                        iconAnchor: [12, 12]
                    });

                    L.marker(latlng, { icon: stopIcon })
                        .addTo(this.mapLayer)
                        .bindPopup(`
                            <div style="font-family: 'Inter', sans-serif; padding: 8px; min-width: 180px;">
                                <div style="display:flex; align-items:center; gap:8px; padding-bottom:8px; border-bottom:1px solid #eee; margin-bottom:8px;">
                                    <div style="
                                        width:28px; height:28px;
                                        background: ${colorScheme.main};
                                        border-radius:50%;
                                        display:flex; align-items:center; justify-content:center;
                                        color:#fff; font-weight:bold; font-size:12px;
                                    ">${idx + 1}</div>
                                    <div>
                                        <div style="font-weight:600; color:#333;">Stop ${idx + 1}</div>
                                        <div style="font-size:11px; color:#666;">${driverName}</div>
                                    </div>
                                </div>
                                <div style="font-size:12px; color:#555; margin-bottom:4px;">
                                    📍 ${stop.address || 'Delivery Location'}
                                </div>
                            </div>
                        `);
                });

                // Draw route line with glow
                L.polyline(latlngs, {
                    color: colorScheme.glow,
                    weight: 8,
                    opacity: 0.3,
                    lineCap: 'round',
                    lineJoin: 'round'
                }).addTo(this.mapLayer);

                // Main line
                L.polyline(latlngs, {
                    color: colorScheme.main,
                    weight: 4,
                    opacity: 0.9,
                    lineCap: 'round',
                    lineJoin: 'round'
                }).addTo(this.mapLayer);

                renderedRoutes.push({
                    color: colorScheme.main,
                    name: driverName,
                    stops: route.stops.length
                });
            }
        });

        // Add legend with correct driver names
        this.addMapLegend(renderedRoutes);

        // Fit bounds
        if (this.mapLayer.getLayers().length > 0) {
            try {
                const group = L.featureGroup(this.mapLayer.getLayers());
                this.map.fitBounds(group.getBounds().pad(0.1));
            } catch (e) {
                // Ignore bounds error
            }
        }

        console.log('Map updated with', renderedRoutes.length, 'routes');
    }

    /**
     * Connect to Server-Sent Events for real-time agent status updates
     */
    connectSSE() {
        // Close existing connection if any
        if (this.eventSource) {
            this.eventSource.close();
        }

        // Only connect if we have a run_id - otherwise no events to subscribe to
        if (!this.currentRunId) {
            console.log('No run_id present, skipping SSE connection');
            return;
        }

        // Use run-scoped SSE endpoint
        const sseUrl = `http://localhost:8000/api/v1/runs/${encodeURIComponent(this.currentRunId)}/agent-events`;

        try {
            this.eventSource = new EventSource(sseUrl);

            this.eventSource.onopen = () => {
                console.log('SSE connection established for run:', this.currentRunId);
            };

            this.eventSource.onmessage = (event) => {
                try {
                    const eventData = JSON.parse(event.data);
                    this.handleAgentEvent(eventData);
                } catch (err) {
                    console.warn('Failed to parse SSE event:', err);
                }
            };

            this.eventSource.onerror = (error) => {
                console.warn('SSE connection error, will retry...', error);
            };
        } catch (error) {
            console.error('Failed to create SSE connection:', error);
        }
    }

    /**
     * Handle incoming agent event from SSE stream.
     * Maps backend agent_name to node ID and applies state machine transitions.
     */
    handleAgentEvent(event) {
        // Skip system/connection events
        if (event.type === 'connected') {
            console.log('[Dashboard] SSE connected:', event.message);
            return;
        }

        const { agent_name, state: eventState, step_type, payload } = event;

        // Map agent name to node ID
        const nodeId = this.agentNameToNodeId[agent_name];
        if (!nodeId) {
            console.log('[Dashboard] Unknown agent, ignoring:', agent_name);
            return;
        }

        // Map event state to visual state
        let visualState = 'idle';
        if (eventState === 'STARTED') {
            visualState = 'running';
        } else if (eventState === 'COMPLETED') {
            visualState = 'completed';
        } else if (eventState === 'ERROR') {
            visualState = 'error';
        }

        // Apply state machine transition
        this.applyAgentState(nodeId, visualState, step_type, payload);

        console.log(`[Dashboard] ${agent_name} → ${eventState} (${nodeId})`);
    }

    /**
     * Apply a state machine state to an agent node.
     * States: idle, running, completed, error.
     * Transitions are persistent — no auto-reset.
     */
    applyAgentState(nodeId, state, stepType, payload) {
        const node = this.nodes.get(nodeId);
        if (!node) return;

        const stateConfig = {
            idle:      { label: 'Idle',      icon: '○', statusClass: 'idle' },
            running:   { label: 'Running',   icon: '⟳', statusClass: 'processing' },
            completed: { label: 'Completed', icon: '✓', statusClass: 'active' },
            error:     { label: 'Error',     icon: '✗', statusClass: 'error' },
        };

        const cfg = stateConfig[state] || stateConfig.idle;

        // Update status badge
        const statusBadge = node.element.querySelector('.agent-status');
        if (statusBadge) {
            statusBadge.classList.remove('active', 'processing', 'idle', 'error');
            statusBadge.classList.add(cfg.statusClass);
            statusBadge.innerHTML = `<span class="dot"></span> ${cfg.icon} ${cfg.label}`;
        }

        // Remove all state classes, apply current
        node.element.classList.remove(
            'agent-idle', 'agent-running', 'agent-completed', 'agent-error',
            'node-active-pulse', 'node-processing-pulse'
        );
        node.element.classList.add(`agent-${state}`);

        // Store state for tracking
        node._agentState = state;
    }

    /**
     * Inject completion tick and error icon badges into all agent nodes.
     * Called once during init.
     */
    injectStateBadges() {
        this.nodes.forEach((node) => {
            if (!node.element.querySelector('.agent-completion-tick')) {
                const tick = document.createElement('span');
                tick.className = 'agent-completion-tick';
                tick.textContent = '✓';
                node.element.appendChild(tick);
            }
            if (!node.element.querySelector('.agent-error-icon')) {
                const err = document.createElement('span');
                err.className = 'agent-error-icon';
                err.textContent = '✗';
                node.element.appendChild(err);
            }
        });
    }

    /**
     * Render agent nodes
     */
    renderAgents(agents) {
        this.nodesContainer.innerHTML = '';

        agents.forEach(agent => {
            const node = this.createAgentNode(agent);
            this.nodesContainer.appendChild(node);
            this.nodes.set(agent.id, {
                element: node,
                data: agent,
                position: this.layout[agent.id] || { x: 100, y: 100 }
            });

            // Position the node
            this.updateNodePosition(agent.id);
        });
    }

    /**
     * Create an agent node element
     */
    createAgentNode(agent) {
        const node = document.createElement('div');
        node.className = `agent-node ${agent.type}`;
        node.dataset.id = agent.id;

        // Status class
        if (agent.status) {
            node.classList.add(agent.status);
        }

        // Icon based on type
        const iconSVG = this.getAgentIcon(agent.type);

        // Processing badge for processing status
        const processingBadge = agent.status === 'processing'
            ? '<span class="processing-badge">Processing</span>'
            : '';

        node.innerHTML = `
            <div class="agent-icon">
                ${iconSVG}
            </div>
            <div class="agent-name">${agent.name}</div>
            <div class="agent-description">${agent.description}</div>
            <div class="agent-footer">
                <div class="agent-status ${agent.status}">
                    <span class="dot"></span>
                    ${this.formatStatus(agent.status)}
                </div>
                ${agent.meta ? `<div class="agent-meta">${agent.meta}</div>` : ''}
                ${processingBadge}
            </div>
        `;

        return node;
    }

    /**
     * Get icon SVG for agent type
     */
    getAgentIcon(type) {
        const icons = {
            orchestrator: `<svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
            </svg>`,
            database: `<svg viewBox="0 0 24 24" fill="currentColor">
                <ellipse cx="12" cy="5" rx="8" ry="3"/>
                <path d="M4 5v6c0 1.66 3.58 3 8 3s8-1.34 8-3V5"/>
                <path d="M4 11v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6"/>
            </svg>`,
            agent: `<svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
            </svg>`
        };

        return icons[type] || icons.agent;
    }

    /**
     * Format status text
     */
    formatStatus(status) {
        const statusMap = {
            active: 'Active',
            processing: 'Processing',
            idle: 'Idle',
            error: 'Error'
        };
        return statusMap[status] || status;
    }

    /**
     * Update node position
     */
    updateNodePosition(nodeId) {
        const node = this.nodes.get(nodeId);
        if (!node) return;

        const pos = node.position;
        node.element.style.left = `${pos.x}px`;
        node.element.style.top = `${pos.y}px`;
    }

    /**
     * Update agent nodes with new status data (called during polling)
     */
    updateAgentStatuses(agents) {
        if (!agents || !Array.isArray(agents)) return;

        agents.forEach(agent => {
            const node = this.nodes.get(agent.id);
            if (!node) return;

            // Update status badge
            const statusBadge = node.element.querySelector('.agent-status');
            if (statusBadge) {
                // Remove all status classes
                statusBadge.classList.remove('active', 'processing', 'idle', 'error');

                // Add current status class
                const status = agent.status || 'idle';
                statusBadge.classList.add(status);

                // Update status text with icon
                const statusText = status.charAt(0).toUpperCase() + status.slice(1);
                const statusIcons = {
                    active: '✓',
                    processing: '⟳',
                    idle: '○',
                    error: '✗'
                };
                statusBadge.innerHTML = `<span class="dot"></span> ${statusIcons[status] || '○'} ${statusText}`;
            }

            // Update description if provided
            const descEl = node.element.querySelector('.node-description');
            if (descEl && agent.description) {
                descEl.textContent = agent.description;
            }

            // Update meta badge if provided
            const metaBadge = node.element.querySelector('.meta-badge');
            if (metaBadge && agent.meta) {
                metaBadge.textContent = agent.meta;
                metaBadge.style.display = 'inline-block';
            }

            // Add visual pulse effect for active agents
            node.element.classList.remove('node-active-pulse', 'node-processing-pulse');
            if (agent.status === 'active') {
                node.element.classList.add('node-active-pulse');
            } else if (agent.status === 'processing') {
                node.element.classList.add('node-processing-pulse');
            }
        });
    }

    /**
     * Draw connections between nodes
     */
    updateConnections(connections) {
        this.connections = connections;
        this.drawConnections();
    }

    /**
     * Draw all connection lines
     */
    drawConnections() {
        // Clear existing connections (keep defs)
        const defs = this.connectionsLayer.querySelector('defs');
        this.connectionsLayer.innerHTML = '';
        this.connectionsLayer.appendChild(defs);

        this.connections.forEach(conn => {
            const fromNode = this.nodes.get(conn.from);
            const toNode = this.nodes.get(conn.to);

            if (!fromNode || !toNode) return;

            // Get node centers
            const fromRect = fromNode.element.getBoundingClientRect();
            const toRect = toNode.element.getBoundingClientRect();
            const containerRect = this.canvasContainer.getBoundingClientRect();

            // Calculate positions relative to container
            const fromX = fromRect.left - containerRect.left + fromRect.width / 2;
            const fromY = fromRect.top - containerRect.top + fromRect.height;
            const toX = toRect.left - containerRect.left + toRect.width / 2;
            const toY = toRect.top - containerRect.top;

            // Create curved path
            const path = this.createConnectionPath(fromX, fromY, toX, toY, conn.active);
            this.connectionsLayer.appendChild(path);

            // Add click handler for data payload
            path.addEventListener('click', (e) => {
                e.stopPropagation();
                this.showDataPayload(conn);
            });
        });
    }

    /**
     * Create a curved SVG path for connection
     */
    createConnectionPath(x1, y1, x2, y2, isActive) {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');

        // Calculate control points for bezier curve
        const midY = (y1 + y2) / 2;
        const controlOffset = Math.abs(y2 - y1) * 0.5;

        const d = `M ${x1} ${y1} 
                   C ${x1} ${y1 + controlOffset}, 
                     ${x2} ${y2 - controlOffset}, 
                     ${x2} ${y2}`;

        path.setAttribute('d', d);
        path.setAttribute('class', isActive ? 'connection-line-animated' : 'connection-line');
        path.style.pointerEvents = 'stroke';
        path.style.cursor = 'pointer';

        return path;
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Node click handler
        this.nodesContainer.addEventListener('click', (e) => {
            const node = e.target.closest('.agent-node');
            if (node) {
                this.openTerminal(node.dataset.id);
            }
        });

        // Node drag handlers
        this.nodesContainer.addEventListener('mousedown', (e) => {
            const node = e.target.closest('.agent-node');
            if (node) {
                this.startDrag(node, e);
            }
        });

        document.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                this.handleDrag(e);
            }
        });

        document.addEventListener('mouseup', () => {
            if (this.isDragging) {
                this.endDrag();
            }
        });

        // Button handlers
        document.getElementById('saveLayoutBtn').addEventListener('click', () => {
            this.saveLayout();
        });

        document.getElementById('resetBtn').addEventListener('click', () => {
            this.resetLayout();
        });

        // Start Allocation handler
        document.getElementById('startAllocationBtn').addEventListener('click', () => {
            this.handleStartAllocation();
        });

        // Map View Toggles
        document.getElementById('graphViewBtn').addEventListener('click', () => {
            this.toggleView('graph');
        });
        document.getElementById('mapViewBtn').addEventListener('click', () => {
            this.toggleView('map');
        });

        // Date Picker
        const dateInput = document.getElementById('historyDate');
        // Set today as default - use LOCAL timezone, not UTC
        const now = new Date();
        const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
        console.log('Initializing date picker with today (local):', today);
        dateInput.value = today;
        API.setSelectedDate(today); // Sync with API

        // Store current date for reference
        this._currentDisplayDate = today;

        dateInput.addEventListener('change', (e) => {
            const newDate = e.target.value;
            console.log('Date picker changed to:', newDate);
            this._currentDisplayDate = newDate;
            API.setSelectedDate(newDate); // Sync with API
            this.loadHistory(newDate);
        });

        // Zoom handlers
        document.getElementById('zoomIn').addEventListener('click', () => {
            this.setZoom(this.zoom + 0.1);
        });

        document.getElementById('zoomOut').addEventListener('click', () => {
            this.setZoom(this.zoom - 0.1);
        });

        // Modal close handlers
        document.getElementById('closeTerminal').addEventListener('click', () => {
            this.closeModal('terminalModal');
        });

        document.getElementById('closeData').addEventListener('click', () => {
            this.closeModal('dataModal');
        });

        // Close modal on backdrop click
        document.querySelectorAll('.modal').forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    modal.classList.remove('active');
                }
            });
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.drawConnections();
            this.updateMinimap();
        });
    }

    /**
     * Start dragging a node
     */
    startDrag(nodeElement, e) {
        const nodeId = nodeElement.dataset.id;
        const node = this.nodes.get(nodeId);

        if (!node) return;

        this.isDragging = true;
        this.dragNode = node;

        const rect = nodeElement.getBoundingClientRect();
        this.dragOffset = {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };

        nodeElement.classList.add('dragging');
    }

    /**
     * Handle drag movement
     */
    handleDrag(e) {
        if (!this.dragNode) return;

        const containerRect = this.canvasContainer.getBoundingClientRect();

        const newX = e.clientX - containerRect.left - this.dragOffset.x;
        const newY = e.clientY - containerRect.top - this.dragOffset.y;

        this.dragNode.position = {
            x: Math.max(0, newX),
            y: Math.max(0, newY)
        };

        // Update node visual position
        this.dragNode.element.style.left = `${this.dragNode.position.x}px`;
        this.dragNode.element.style.top = `${this.dragNode.position.y}px`;

        // Redraw connections
        this.drawConnections();

        // Update minimap
        this.updateMinimap();
    }

    /**
     * End dragging
     */
    endDrag() {
        if (this.dragNode) {
            this.dragNode.element.classList.remove('dragging');

            // Update layout
            this.layout[this.dragNode.data.id] = this.dragNode.position;
        }

        this.isDragging = false;
        this.dragNode = null;
    }

    /**
     * Open terminal modal for an agent
     */
    openTerminal(nodeId) {
        if (this.isDragging) return;

        const node = this.nodes.get(nodeId);
        if (!node) return;

        const modal = document.getElementById('terminalModal');
        const title = document.getElementById('terminalTitle');
        const output = document.getElementById('terminalOutput');

        title.textContent = `${node.data.name} - Terminal`;

        // Try to get real agent logs first
        const realLogs = API.getAgentLogs(nodeId);

        if (realLogs && realLogs.length > 0) {
            output.textContent = this.formatRealLogs(realLogs, node.data);
        } else {
            // Fallback to mock terminal output
            output.textContent = this.generateTerminalOutput(node.data);
        }

        modal.classList.add('active');
    }

    /**
     * Format real agent logs for terminal display
     */
    formatRealLogs(logs, agent) {
        let output = `=== ${agent.name} Decision Log ===\n\n`;

        logs.forEach((log, index) => {
            const timestamp = log.timestamp || new Date().toISOString();
            const stepType = log.step_type || log.action || 'Decision';

            output += `[${timestamp}] Step ${index + 1}: ${stepType}\n`;

            // Show short message if available
            if (log.short_message) {
                output += `  Message: ${log.short_message}\n`;
            }

            // Show details if available
            if (log.details && Object.keys(log.details).length > 0) {
                output += `  Details:\n`;
                Object.entries(log.details).forEach(([key, value]) => {
                    let displayValue;
                    if (typeof value === 'number') {
                        displayValue = Number.isInteger(value) ? value : value.toFixed(2);
                    } else if (typeof value === 'object') {
                        displayValue = JSON.stringify(value);
                    } else {
                        displayValue = value;
                    }
                    // Format key to be more readable
                    const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                    output += `    - ${formattedKey}: ${displayValue}\n`;
                });
            }

            // Legacy fields support
            if (log.decision) {
                output += `  Decision: ${log.decision}\n`;
            }
            if (log.reasoning) {
                output += `  Reasoning: ${log.reasoning}\n`;
            }
            if (log.input_snapshot) {
                const summary = JSON.stringify(log.input_snapshot).substring(0, 150);
                output += `  Input: ${summary}${summary.length >= 150 ? '...' : ''}\n`;
            }
            if (log.output_snapshot) {
                const summary = JSON.stringify(log.output_snapshot).substring(0, 150);
                output += `  Output: ${summary}${summary.length >= 150 ? '...' : ''}\n`;
            }
            output += '\n';
        });

        output += `=== Total: ${logs.length} decision(s) ===`;
        return output;
    }

    /**
     * Generate mock terminal output
     */
    generateTerminalOutput(agent) {
        const timestamp = new Date().toISOString();
        const allocationRun = API.getCurrentAllocationRun();

        // If we have a real allocation run, show real info
        if (allocationRun) {
            const outputs = {
                central_orchestrator: `[${timestamp}] Central Orchestrator Agent - LIVE\n[${timestamp}] Current Allocation Run: ${allocationRun.id.substring(0, 8)}...\n[${timestamp}] Status: ${allocationRun.status || 'COMPLETED'}\n[${timestamp}] Gini Index: ${allocationRun.gini_index?.toFixed(4) || 'N/A'}\n[${timestamp}] Std Dev: ${allocationRun.std_dev?.toFixed(2) || 'N/A'}\n[${timestamp}] Avg Workload: ${allocationRun.avg_workload?.toFixed(2) || 'N/A'}`,

                route_database: `[${timestamp}] Route Database Connection - LIVE\n[${timestamp}] Host: localhost (SQLite)\n[${timestamp}] Database: fair_dispatch.db\n[${timestamp}] Allocation Run ID: ${allocationRun.id}\n[${timestamp}] Last sync: ${allocationRun.created_at || timestamp}\n[${timestamp}] Connection: Active`
            };

            if (outputs[agent.id]) {
                return outputs[agent.id];
            }
        }

        // Default mock outputs
        const mockOutputs = {
            central_orchestrator: `[${timestamp}] Central Orchestrator Agent initialized\n[${timestamp}] Monitoring 6 child agents\n[${timestamp}] Data flow channels: ACTIVE\n[${timestamp}] Current task: Coordinating route allocation\n[${timestamp}] Status: All systems nominal\n[${timestamp}] Last heartbeat: 2ms ago`,

            route_database: `[${timestamp}] Route Database Connection\n[${timestamp}] Host: localhost\n[${timestamp}] Database: fair_dispatch\n[${timestamp}] Last sync: ${timestamp}\n[${timestamp}] Connection pool: healthy`,

            route_planner: `[${timestamp}] Route Planner Agent - Standby\n[${timestamp}] Awaiting allocation request...\n[${timestamp}] K-Means clustering: Ready\n[${timestamp}] OR-Tools optimization: Ready`,

            fairness_manager: `[${timestamp}] Fairness Manager Agent\n[${timestamp}] Gini threshold: 0.30\n[${timestamp}] Awaiting route proposals...`,

            driver_liaison: `[${timestamp}] Driver Liaison Agent\n[${timestamp}] Processing driver feedback...\n[${timestamp}] Driver contexts loaded\n[${timestamp}] Awaiting route proposals...`,

            explainability: `[${timestamp}] Explainability Agent\n[${timestamp}] Template engine: ACTIVE\n[${timestamp}] Language support: en, ta, hi\n[${timestamp}] Gemini integration: READY`,

            learning: `[${timestamp}] Learning Agent\n[${timestamp}] Model: Ready\n[${timestamp}] Mode: Online learning`
        };

        return mockOutputs[agent.id] || `[${timestamp}] Agent ${agent.name} initialized`;
    }

    /**
     * Show data payload modal
     */
    showDataPayload(connection) {
        const modal = document.getElementById('dataModal');
        const title = document.getElementById('dataTitle');
        const payload = document.getElementById('dataPayload');

        const fromNode = this.nodes.get(connection.from);
        const toNode = this.nodes.get(connection.to);

        title.textContent = `Data Flow: ${fromNode?.data.name || connection.from} → ${toNode?.data.name || connection.to}`;

        // Generate mock payload
        payload.textContent = JSON.stringify(this.generateDataPayload(connection), null, 2);

        modal.classList.add('active');
    }

    /**
     * Generate mock data payload
     */
    generateDataPayload(connection) {
        const payloads = {
            'central_orchestrator-route_planner': {
                type: 'allocation_request',
                timestamp: new Date().toISOString(),
                packages: 47,
                drivers: 5,
                warehouse: { lat: 12.9716, lng: 77.5946 },
                config: {
                    target_packages_per_route: 20,
                    fairness_threshold: 0.30
                }
            },
            'route_planner-fairness_manager': {
                type: 'route_proposal',
                proposal_id: 1,
                assignments: [
                    { driver_id: 'driver_001', route_id: 'route_001', effort: 65.3 },
                    { driver_id: 'driver_002', route_id: 'route_002', effort: 72.1 }
                ],
                total_effort: 324.5
            }
        };

        const key = `${connection.from}-${connection.to}`;
        return payloads[key] || {
            type: 'data_flow',
            from: connection.from,
            to: connection.to,
            active: connection.active,
            timestamp: new Date().toISOString()
        };
    }

    /**
     * Close modal
     */
    closeModal(modalId) {
        document.getElementById(modalId).classList.remove('active');
    }

    /**
     * Update zoom level
     */
    setZoom(level) {
        this.zoom = Math.max(0.5, Math.min(2, level));
        this.nodesContainer.style.transform = `scale(${this.zoom})`;
        this.nodesContainer.style.transformOrigin = 'center center';
        this.updateMinimap();
    }

    /**
     * Save layout to localStorage
     */
    saveLayout() {
        const layoutData = {};
        this.nodes.forEach((node, id) => {
            layoutData[id] = node.position;
        });

        localStorage.setItem('agentDashboardLayout', JSON.stringify(layoutData));

        // Show brief confirmation
        this.showToast('Layout saved!');
    }

    /**
     * Load layout from localStorage
     */
    loadLayout() {
        try {
            const saved = localStorage.getItem('agentDashboardLayout');
            if (saved) {
                const layoutData = JSON.parse(saved);
                this.layout = { ...this.getDefaultLayout(), ...layoutData };
            }
        } catch (e) {
            console.warn('Could not load saved layout:', e);
        }
    }

    /**
     * Reset layout to default
     */
    resetLayout() {
        this.layout = this.getDefaultLayout();
        localStorage.removeItem('agentDashboardLayout');

        // Update all node positions
        this.nodes.forEach((node, id) => {
            node.position = this.layout[id] || { x: 100, y: 100 };
            this.updateNodePosition(id);
        });

        // Redraw connections
        this.drawConnections();
        this.updateMinimap();

        this.showToast('Layout reset!');
    }

    /**
     * Show toast notification
     */
    showToast(message) {
        // Simple toast implementation
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed;
            bottom: 60px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--accent-primary);
            color: var(--bg-primary);
            padding: 10px 20px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            z-index: 2000;
            animation: fadeInOut 2s ease;
        `;
        toast.textContent = message;
        document.body.appendChild(toast);

        setTimeout(() => toast.remove(), 2000);
    }

    /**
     * Update minimap
     */
    updateMinimap() {
        // Clear minimap
        this.minimap.innerHTML = '<div class="minimap-viewport" id="minimapViewport"></div>';

        const minimapWidth = 180;
        const minimapHeight = 120;
        const containerWidth = this.canvasContainer.clientWidth;
        const containerHeight = this.canvasContainer.clientHeight;

        const scaleX = minimapWidth / containerWidth;
        const scaleY = minimapHeight / containerHeight;
        const scale = Math.min(scaleX, scaleY);

        // Draw nodes on minimap
        this.nodes.forEach((node) => {
            const miniNode = document.createElement('div');
            miniNode.className = 'minimap-node';
            miniNode.style.left = `${node.position.x * scale}px`;
            miniNode.style.top = `${node.position.y * scale}px`;
            miniNode.style.width = '8px';
            miniNode.style.height = '5px';
            this.minimap.appendChild(miniNode);
        });

        // Update viewport indicator
        const viewport = document.getElementById('minimapViewport');
        if (viewport) {
            viewport.style.width = `${minimapWidth}px`;
            viewport.style.height = `${minimapHeight}px`;
            viewport.style.left = '0';
            viewport.style.top = '0';
        }
    }

    /**
     * Update stats display
     */
    updateStats(stats) {
        document.getElementById('processingCount').textContent = stats.processing;
        document.getElementById('dataFlowCount').textContent = stats.dataFlows;
        document.getElementById('agentCount').textContent = stats.totalAgents;
    }

    /**
     * Setup API polling for real-time updates
     */
    setupAPIPolling() {
        // Handle connection status updates
        API.onStatusUpdate((status) => {
            // Update connection indicator
            const liveIndicator = document.querySelector('.status-indicator');
            const liveStatus = document.getElementById('liveStatus');

            if (status.connected) {
                if (liveStatus) liveStatus.textContent = 'Live';
                if (liveIndicator) {
                    liveIndicator.classList.remove('disconnected');
                    liveIndicator.querySelector('.status-dot').style.background = '';
                }
            } else {
                if (liveStatus) liveStatus.textContent = 'Offline';
                if (liveIndicator) {
                    liveIndicator.classList.add('disconnected');
                    liveIndicator.querySelector('.status-dot').style.background = 'var(--status-error)';
                }
            }
        });

        // Handle workflow updates (real-time data)
        API.onWorkflowUpdate((state) => {
            // Debug log
            console.log(`[${new Date().toLocaleTimeString()}] Poll update received. Real data: ${state.isRealData}, Run ID: ${state.allocationRun?.id?.substring(0, 8) || 'none'}`);

            if (state && state.isRealData !== undefined) {
                // Check if this is new data
                const isNewData = !this.currentWorkflowState ||
                    (state.allocationRun?.id !== this.currentWorkflowState?.allocationRun?.id);

                // Update current state
                this.currentWorkflowState = state;

                // Re-render agents with new data - ALWAYS update to show real statuses
                this.updateAgentStatuses(state.agents);

                // Update connections - ALWAYS update to show active flows  
                this.updateConnections(state.connections);

                // Update stats
                this.updateStats(state.stats);

                // Update minimap
                this.updateMinimap();

                // Show notification if new allocation (but don't change date picker)
                if (isNewData && state.isRealData) {
                    this.showToast(`🚀 New allocation! Run: ${state.allocationRun.id.substring(0, 8)}...`);

                    // Refresh map if active
                    if (document.getElementById('mapWrapper')?.classList.contains('active')) {
                        this.renderMap(state.allocationRun);
                    }

                    // Trigger Replay Animation for the workflow
                    if (state.timeline) {
                        this.playWorkflowAnimation(state.timeline);
                    }
                }

                // Also update map in background if we have allocation data
                if (state.allocationRun && this.map) {
                    this.renderMap(state.allocationRun);
                }
            }
        });

        API.onError((error) => {
            console.error('API Error:', error);
        });

        // Start polling
        API.startPolling();
    }

    /**
     * Simulate agent activity for demo
     */
    simulateActivity() {
        const agentIds = ['route_planner', 'fairness_manager', 'driver_liaison', 'explainability', 'learning'];
        let currentIndex = 0;

        setInterval(() => {
            // Reset all to idle
            agentIds.forEach(id => {
                const node = this.nodes.get(id);
                if (node) {
                    node.element.classList.remove('processing');
                    node.data.status = 'idle';
                }
            });

            // Set current to processing
            const currentId = agentIds[currentIndex];
            const currentNode = this.nodes.get(currentId);
            if (currentNode) {
                currentNode.element.classList.add('processing');
                currentNode.data.status = 'processing';
            }

            // Update connections
            const state = API.getMockWorkflowState();
            state.connections.forEach((conn, i) => {
                conn.active = i <= currentIndex + 1;
            });
            this.updateConnections(state.connections);

            // Move to next
            currentIndex = (currentIndex + 1) % agentIds.length;

            // Update stats
            this.updateStats({
                processing: 1,
                dataFlows: currentIndex + 2,
                totalAgents: 6
            });
        }, 3000);
    }

    /**
     * Handle start allocation click
     */
    async handleStartAllocation() {
        const btn = document.getElementById('startAllocationBtn');
        if (btn.disabled) return;

        // FIRST: Force update date picker to today immediately (LOCAL timezone)
        const now = new Date();
        const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
        const dateInput = document.getElementById('historyDate');
        if (dateInput) {
            dateInput.value = today;
            this._currentDisplayDate = today;
            console.log('[Start Allocation] Date picker forced to:', today);
        }
        API.setSelectedDate(today);

        // Clear any cached old data
        API._lastKnownRunId = null;
        API._lastRunCount = 0;
        API._currentAllocationRun = null;

        // Disable button and show loading state
        btn.disabled = true;
        const originalText = btn.innerHTML;
        btn.innerHTML = `<span class="status-badge processing" style="background:transparent; border:none; padding:0;"><svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="2" style="animation: spin 1s linear infinite;"><path d="M7 1v2M7 11v2M1 7h2M11 7h2"/></svg></span> Starting...`;

        // Set all agents to "processing" state for visual feedback
        this.setAgentsProcessing();

        this.showToast('🚀 Initiating Agent Workflow with 30 packages & 5 drivers...');

        try {
            // Generate comprehensive test data for Bangalore area
            const payload = this.generateTestAllocationData();

            // Call API
            const result = await API.triggerAllocation(payload);

            // Success
            this.showToast(`🎉 Workflow Complete! Run ID: ${result.allocation_run_id.substring(0, 8)}... | Gini: ${result.global_fairness.gini_index.toFixed(3)}`);
            console.log('Allocation result:', result);

            // Force refresh the dashboard data
            API._lastKnownRunId = null;
            API._lastRunCount = 0;
            API._currentAllocationRun = null;

            const workflowState = await API.getRealWorkflowState(today);
            if (workflowState) {
                // Update UI with new data
                this.currentWorkflowState = workflowState;
                this.updateAgentStatuses(workflowState.agents);
                this.updateConnections(workflowState.connections);
                this.updateStats(workflowState.stats);

                // Render map if allocation run exists
                if (workflowState.allocationRun) {
                    this.renderMap(workflowState.allocationRun);
                }

                // Play workflow animation
                if (workflowState.timeline) {
                    this.playWorkflowAnimation(workflowState.timeline);
                }
            }

        } catch (error) {
            console.error('Allocation start failed:', error);
            this.showToast('❌ Failed to start: ' + error.message);
            // Reset agents to idle on error
            this.resetAgentsToIdle();
        } finally {
            // Restore button after delay
            setTimeout(() => {
                btn.disabled = false;
                btn.innerHTML = originalText;
            }, 2000);
        }
    }

    /**
     * Set all agents to running state using state machine
     */
    setAgentsProcessing() {
        const agentIds = ['route_planner', 'fairness_manager', 'driver_liaison', 'explainability', 'learning'];
        agentIds.forEach(id => {
            this.applyAgentState(id, 'running', '', {});
        });
    }

    /**
     * Reset all agents to idle state using state machine
     */
    resetAgentsToIdle() {
        const agentIds = ['route_planner', 'fairness_manager', 'driver_liaison', 'explainability', 'learning'];
        agentIds.forEach(id => {
            this.applyAgentState(id, 'idle', '', {});
        });
    }

    /**
     * Mark all agents as completed using state machine
     */
    completeAllAgents() {
        const allIds = ['central_orchestrator', 'route_database', 'route_planner', 'fairness_manager', 'driver_liaison', 'explainability', 'learning'];
        allIds.forEach(id => {
            const node = this.nodes.get(id);
            if (node && node._agentState !== 'completed' && node._agentState !== 'error') {
                this.applyAgentState(id, 'completed', '', {});
            }
        });
    }

    /**
     * Generate comprehensive test allocation data
     * Creates realistic packages and drivers for Bangalore area
     */
    generateTestAllocationData() {
        const timestamp = Date.now();

        // Bangalore area locations with realistic addresses
        const locations = [
            { name: "Electronic City Phase 1", lat: 12.8456, lng: 77.6603, area: "South" },
            { name: "Electronic City Phase 2", lat: 12.8520, lng: 77.6680, area: "South" },
            { name: "Whitefield Main Road", lat: 12.9698, lng: 77.7500, area: "East" },
            { name: "ITPL Main Road", lat: 12.9854, lng: 77.7081, area: "East" },
            { name: "Marathahalli Bridge", lat: 12.9591, lng: 77.6974, area: "East" },
            { name: "Indiranagar 100ft Road", lat: 12.9784, lng: 77.6408, area: "Central" },
            { name: "Indiranagar CMH Road", lat: 12.9716, lng: 77.6412, area: "Central" },
            { name: "Koramangala 4th Block", lat: 12.9352, lng: 77.6245, area: "Central" },
            { name: "Koramangala 5th Block", lat: 12.9344, lng: 77.6150, area: "Central" },
            { name: "HSR Layout Sector 1", lat: 12.9116, lng: 77.6389, area: "South" },
            { name: "HSR Layout Sector 7", lat: 12.9081, lng: 77.6476, area: "South" },
            { name: "BTM Layout 1st Stage", lat: 12.9166, lng: 77.6101, area: "South" },
            { name: "JP Nagar 5th Phase", lat: 12.9063, lng: 77.5857, area: "South" },
            { name: "Jayanagar 4th Block", lat: 12.9308, lng: 77.5838, area: "South" },
            { name: "Bannerghatta Road", lat: 12.8876, lng: 77.5973, area: "South" },
            { name: "MG Road", lat: 12.9756, lng: 77.6068, area: "Central" },
            { name: "Brigade Road", lat: 12.9716, lng: 77.6070, area: "Central" },
            { name: "Commercial Street", lat: 12.9824, lng: 77.6074, area: "Central" },
            { name: "Rajajinagar 1st Block", lat: 12.9914, lng: 77.5521, area: "West" },
            { name: "Malleswaram 18th Cross", lat: 13.0067, lng: 77.5713, area: "North" },
            { name: "Yeshwanthpur", lat: 13.0271, lng: 77.5450, area: "North" },
            { name: "Hebbal Flyover", lat: 13.0358, lng: 77.5970, area: "North" },
            { name: "Yelahanka New Town", lat: 13.1007, lng: 77.5963, area: "North" },
            { name: "RT Nagar", lat: 13.0206, lng: 77.5970, area: "North" },
            { name: "Basavanagudi Bull Temple", lat: 12.9429, lng: 77.5688, area: "South" },
            { name: "Vijayanagar BDA Complex", lat: 12.9710, lng: 77.5360, area: "West" },
            { name: "Kengeri Satellite Town", lat: 12.9048, lng: 77.4823, area: "West" },
            { name: "Banashankari 2nd Stage", lat: 12.9255, lng: 77.5468, area: "South" },
            { name: "Sarjapur Road", lat: 12.9107, lng: 77.6868, area: "East" },
            { name: "Bellandur Lake View", lat: 12.9260, lng: 77.6762, area: "East" },
        ];

        // Generate 30 packages with varied properties
        const priorities = ["NORMAL", "NORMAL", "NORMAL", "HIGH", "HIGH", "EXPRESS"];
        const packages = locations.map((loc, index) => ({
            id: `pkg_${timestamp}_${String(index + 1).padStart(3, '0')}`,
            weight_kg: parseFloat((Math.random() * 8 + 0.5).toFixed(2)), // 0.5 to 8.5 kg
            fragility_level: Math.floor(Math.random() * 4) + 1, // 1-4
            address: `${loc.name}, Bangalore - ${560000 + Math.floor(Math.random() * 100)}`,
            latitude: loc.lat + (Math.random() - 0.5) * 0.01, // Small random offset
            longitude: loc.lng + (Math.random() - 0.5) * 0.01,
            priority: priorities[Math.floor(Math.random() * priorities.length)]
        }));

        // Generate 5 drivers with different capacities and languages
        const driverNames = [
            { name: "Rajesh Kumar", lang: "hi", capacity: 200 },
            { name: "Venkatesh S", lang: "kn", capacity: 180 },
            { name: "Mohammed Ashraf", lang: "en", capacity: 220 },
            { name: "Suresh Babu", lang: "ta", capacity: 150 },
            { name: "Prashanth Gowda", lang: "kn", capacity: 250 }
        ];

        const drivers = driverNames.map((d, index) => ({
            id: `driver_${timestamp}_${String(index + 1).padStart(2, '0')}`,
            name: d.name,
            vehicle_capacity_kg: d.capacity,
            preferred_language: d.lang
        }));

        return {
            allocation_date: API.getTodayDate(),
            warehouse: { lat: 12.9716, lng: 77.5946 }, // MG Road area - central Bangalore
            packages,
            drivers
        };
    }

    /**
     * Map Integration Methods
     */
    initMap() {
        if (this.map) return;

        // Initialize Leaflet map
        this.map = L.map('map').setView([12.9716, 77.5946], 12);

        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 19
        }).addTo(this.map);

        this.mapLayer = L.layerGroup().addTo(this.map);
        this.renderMap(this.currentWorkflowState?.allocationRun);
    }

    toggleView(view) {
        const graphBtn = document.getElementById('graphViewBtn');
        const mapBtn = document.getElementById('mapViewBtn');
        const mapWrapper = document.getElementById('mapWrapper');

        if (view === 'map') {
            mapBtn.classList.add('active');
            graphBtn.classList.remove('active');
            mapWrapper.classList.add('active');
            if (!this.map) this.initMap();
        } else {
            graphBtn.classList.add('active');
            mapBtn.classList.remove('active');
            mapWrapper.classList.remove('active');
        }
    }

    async renderMap(allocationRun) {
        if (!this.map || !allocationRun) return;

        this.mapLayer.clearLayers();

        // Premium color palette with gradients
        const routeColors = [
            { main: '#3b82f6', glow: '#60a5fa', name: 'Route A' },
            { main: '#f59e0b', glow: '#fbbf24', name: 'Route B' },
            { main: '#ec4899', glow: '#f472b6', name: 'Route C' },
            { main: '#10b981', glow: '#34d399', name: 'Route D' },
            { main: '#8b5cf6', glow: '#a78bfa', name: 'Route E' },
            { main: '#f97316', glow: '#fb923c', name: 'Route F' }
        ];

        // Premium Warehouse marker with pulsing animation
        const warehouseIcon = L.divIcon({
            className: 'warehouse-marker-premium',
            html: `
                <div style="position:relative;">
                    <div style="
                        position:absolute;
                        width:40px; height:40px;
                        background: radial-gradient(circle, rgba(0,212,170,0.3) 0%, transparent 70%);
                        border-radius:50%;
                        animation: pulse 2s infinite;
                        left:-12px; top:-12px;
                    "></div>
                    <div style="
                        width:20px; height:20px;
                        background: linear-gradient(135deg, #00d4aa 0%, #00a388 100%);
                        border-radius:50%;
                        border:3px solid #fff;
                        box-shadow: 0 0 20px rgba(0,212,170,0.6), 0 4px 8px rgba(0,0,0,0.3);
                        display:flex; align-items:center; justify-content:center;
                    ">
                        <span style="color:#fff; font-size:10px; font-weight:bold;">W</span>
                    </div>
                </div>
            `,
            iconSize: [20, 20],
            iconAnchor: [10, 10]
        });

        L.marker([12.9716, 77.5946], { icon: warehouseIcon })
            .addTo(this.mapLayer)
            .bindPopup(`
                <div style="font-family: 'Inter', sans-serif; padding: 8px;">
                    <div style="font-size:14px; font-weight:600; color:#00d4aa; margin-bottom:4px;">
                        🏭 Central Warehouse
                    </div>
                    <div style="font-size:12px; color:#888;">Bangalore Hub</div>
                    <div style="font-size:11px; color:#666; margin-top:4px;">
                        📍 12.9716°N, 77.5946°E
                    </div>
                </div>
            `);

        // Get assignments
        let assignments = allocationRun.assignments;
        if (!assignments || assignments.length === 0) {
            try {
                const dateStr = allocationRun.run_date || allocationRun.allocation_date || API.getTodayDate();
                const response = await API.getAssignments(dateStr);
                assignments = response?.items || [];
            } catch (e) {
                console.error('Failed to fetch assignments:', e);
                assignments = [];
            }
        }

        if (!assignments || assignments.length === 0) {
            // Add "No routes" indicator
            this.addMapLegend([]);
            return;
        }

        // Render each route with premium styling
        const renderedRoutes = [];
        for (let i = 0; i < assignments.length; i++) {
            const assign = assignments[i];
            try {
                const routeId = assign.route_id || assign.route?.id;
                if (!routeId) continue;

                const route = await API.getRoute(routeId);
                const colorScheme = routeColors[i % routeColors.length];
                const driverName = assign.driver?.name || assign.driver_name || `Driver ${i + 1}`;

                if (route.stops && route.stops.length > 0) {
                    const latlngs = [[12.9716, 77.5946]];

                    // Render stops with premium markers
                    route.stops.forEach((stop, idx) => {
                        const latlng = [stop.latitude, stop.longitude];
                        latlngs.push(latlng);

                        // Premium stop marker with number
                        const stopIcon = L.divIcon({
                            className: 'stop-marker-premium',
                            html: `
                                <div style="
                                    width:24px; height:24px;
                                    background: linear-gradient(135deg, ${colorScheme.main} 0%, ${colorScheme.glow} 100%);
                                    border-radius:50%;
                                    border:2px solid #fff;
                                    box-shadow: 0 2px 8px rgba(0,0,0,0.3), 0 0 12px ${colorScheme.main}40;
                                    display:flex; align-items:center; justify-content:center;
                                    font-size:10px; font-weight:bold; color:#fff;
                                ">${idx + 1}</div>
                            `,
                            iconSize: [24, 24],
                            iconAnchor: [12, 12]
                        });

                        L.marker(latlng, { icon: stopIcon })
                            .addTo(this.mapLayer)
                            .bindPopup(`
                                <div style="font-family: 'Inter', sans-serif; padding: 8px; min-width: 180px;">
                                    <div style="
                                        display:flex; align-items:center; gap:8px;
                                        padding-bottom:8px; border-bottom:1px solid #eee; margin-bottom:8px;
                                    ">
                                        <div style="
                                            width:28px; height:28px;
                                            background: ${colorScheme.main};
                                            border-radius:50%;
                                            display:flex; align-items:center; justify-content:center;
                                            color:#fff; font-weight:bold; font-size:12px;
                                        ">${idx + 1}</div>
                                        <div>
                                            <div style="font-weight:600; color:#333;">Stop ${idx + 1}</div>
                                            <div style="font-size:11px; color:#666;">${driverName}</div>
                                        </div>
                                    </div>
                                    <div style="font-size:12px; color:#555; margin-bottom:4px;">
                                        📍 ${stop.address || 'Delivery Location'}
                                    </div>
                                    <div style="display:flex; gap:12px; font-size:11px; color:#888;">
                                        <span>📦 ${stop.weight_kg?.toFixed(1) || '?'} kg</span>
                                        <span>⚡ ${stop.priority || 'Normal'}</span>
                                    </div>
                                </div>
                            `);
                    });

                    // Animated route line with glow effect
                    // Background glow
                    L.polyline(latlngs, {
                        color: colorScheme.glow,
                        weight: 8,
                        opacity: 0.3,
                        lineCap: 'round',
                        lineJoin: 'round'
                    }).addTo(this.mapLayer);

                    // Main line
                    L.polyline(latlngs, {
                        color: colorScheme.main,
                        weight: 4,
                        opacity: 0.9,
                        lineCap: 'round',
                        lineJoin: 'round'
                    }).addTo(this.mapLayer);

                    // Direction arrows
                    const decorator = L.polylineDecorator ? L.polylineDecorator(L.polyline(latlngs), {
                        patterns: [{
                            offset: 25,
                            repeat: 50,
                            symbol: L.Symbol.arrowHead({
                                pixelSize: 8,
                                polygon: false,
                                pathOptions: { stroke: true, color: '#fff', weight: 2, opacity: 0.8 }
                            })
                        }]
                    }).addTo(this.mapLayer) : null;

                    renderedRoutes.push({
                        color: colorScheme.main,
                        name: driverName,
                        stops: route.stops.length,
                        distance: route.total_distance_km || 'N/A'
                    });
                }
            } catch (e) {
                console.error('Error rendering route:', e);
            }
        }

        // Add legend
        this.addMapLegend(renderedRoutes);

        // Fit bounds to show all markers
        if (this.mapLayer.getLayers().length > 0) {
            try {
                const group = L.featureGroup(this.mapLayer.getLayers());
                this.map.fitBounds(group.getBounds().pad(0.1));
            } catch (e) {
                // Ignore bounds error
            }
        }
    }

    addMapLegend(routes) {
        // Remove existing legend
        if (this.mapLegend) {
            this.map.removeControl(this.mapLegend);
        }

        // Create custom legend control
        const LegendControl = L.Control.extend({
            options: { position: 'bottomright' },
            onAdd: function () {
                const div = L.DomUtil.create('div', 'map-legend');
                div.style.cssText = `
                    background: rgba(15, 23, 42, 0.95);
                    backdrop-filter: blur(10px);
                    padding: 12px 16px;
                    border-radius: 12px;
                    border: 1px solid rgba(255,255,255,0.1);
                    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                    font-family: 'Inter', sans-serif;
                    min-width: 160px;
                `;

                let html = `<div style="font-size:12px; font-weight:600; color:#fff; margin-bottom:8px; display:flex; align-items:center; gap:6px;">
                    <span style="font-size:14px;">🗺️</span> Route Legend
                </div>`;

                if (routes.length === 0) {
                    html += `<div style="font-size:11px; color:#888;">No routes to display</div>`;
                } else {
                    routes.forEach((route, i) => {
                        html += `
                            <div style="display:flex; align-items:center; gap:8px; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                                <div style="width:12px; height:12px; background:${route.color}; border-radius:50%;"></div>
                                <div style="flex:1;">
                                    <div style="font-size:11px; color:#fff; font-weight:500;">${route.name}</div>
                                    <div style="font-size:10px; color:#888;">${route.stops} stops</div>
                                </div>
                            </div>
                        `;
                    });
                }

                div.innerHTML = html;
                return div;
            }
        });

        this.mapLegend = new LegendControl();
        this.map.addControl(this.mapLegend);
    }

    async loadHistory(date) {
        this.showToast(`Loading history for ${date}...`);
        try {
            const response = await API.getAllocationRuns(date);
            const runs = response?.runs || [];
            if (runs && runs.length > 0) {
                // Load the first run of the day for now
                const latestRun = runs[runs.length - 1]; // Get the latest

                // Construct a state object similar to verify/real data
                // Note: We need to manually construct the state structure as getRealWorkflowState does
                // Or refactor to reuse that logic. For now, simple update.

                this.currentWorkflowState = {
                    allocationRun: latestRun,
                    isRealData: true,
                    agents: {}, // We might miss agent logs for history unless we fetch timeline too
                    stats: {},
                    connections: []
                };

                // Fetch timeline for logs
                const timeline = await API.getAgentTimeline(latestRun.id);
                const state = API.transformTimelineToWorkflowState(timeline, latestRun);

                this.currentWorkflowState = state;

                // Update UI
                this.updateAgentStatuses(state.agents);
                this.updateStats(state.stats);
                this.updateConnections(state.connections);
                this.renderMap(latestRun);

                this.showToast(`Loaded ${runs.length} runs for ${date}`);
            } else {
                this.showToast(`No data found for ${date}`);
            }
        } catch (e) {
            console.error(e);
            this.showToast('Failed to load history');
        }
    }


    /**
     * Animate the workflow steps
     */
    async playWorkflowAnimation(timeline) {
        // Handle both new format (timeline.timeline) and legacy format (timeline.steps)
        const steps = timeline?.timeline || timeline?.steps || [];
        if (!steps || steps.length === 0) return;

        this.showToast('Replaying agent workflow...');
        console.log('Starting workflow replay with steps:', steps.length);

        // Helper to reset all nodes visual state
        this.nodes.forEach(node => {
            // Remove status classes
            const statusBadge = node.element.querySelector('.agent-status');
            if (statusBadge) {
                statusBadge.className = 'agent-status idle';
                statusBadge.innerHTML = '<span class="dot"></span> Idle';
            }
            node.element.classList.remove('active', 'processing');
        });

        // Map backend agent names to frontend node IDs
        const agentMap = {
            'MLEffortAgent': 'route_planner',
            'RoutePlannerAgent': 'route_planner',
            'FairnessManagerAgent': 'fairness_manager',
            'DriverLiaisonAgent': 'driver_liaison',
            'ExplainabilityAgent': 'explainability',
            'LearningAgent': 'learning',
        };

        for (const step of steps) {
            const nodeId = agentMap[step.agent_name];
            if (!nodeId) continue;

            const node = this.nodes.get(nodeId);
            if (node) {
                // Update to processing
                const statusBadge = node.element.querySelector('.agent-status');
                if (statusBadge) {
                    statusBadge.className = 'agent-status processing';
                    statusBadge.innerHTML = '<span class="dot"></span> Processing';
                }
                node.element.classList.add('processing');

                // Log to terminal
                const logMsg = `[${step.agent_name}] ${step.step_type}`;
                console.log(logMsg);

                // Use the existing terminal if open, or update the mock one
                // We'll assuming updateConnections highlights the flow too

                await new Promise(r => setTimeout(r, 1200)); // Visible delay

                // Update to active/done
                if (statusBadge) {
                    statusBadge.className = 'agent-status active';
                    statusBadge.innerHTML = '<span class="dot"></span> Active';
                }
                node.element.classList.remove('processing');
                node.element.classList.add('active');
            }
        }

        this.showToast('Workflow Replay Complete');
    }
}

// Add fadeInOut animation
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeInOut {
        0% { opacity: 0; transform: translateX(-50%) translateY(20px); }
        20% { opacity: 1; transform: translateX(-50%) translateY(0); }
        80% { opacity: 1; transform: translateX(-50%) translateY(0); }
        100% { opacity: 0; transform: translateX(-50%) translateY(-20px); }
    }
`;
document.head.appendChild(style);

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new AgentSupervisionDashboard();

    // Start activity simulation for demo
    // window.dashboard.simulateActivity();
});

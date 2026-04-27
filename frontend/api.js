/**
 * API Integration Module
 * Handles communication with the Fair Dispatch backend
 * Real-time data fetching with polling support
 */

const API = {
    BASE_URL: 'http://localhost:8000',
    API_PREFIX: '/api/v1',

    // Polling interval in milliseconds - increased to reduce server load
    POLL_INTERVAL: 10000, // 10 seconds for routine checks
    FAST_POLL_INTERVAL: 3000, // 3 seconds when actively monitoring

    // Current polling timer
    _pollTimer: null,

    // Cache for current allocation run
    _currentAllocationRun: null,
    _lastAgentTimeline: null,
    
    // Track last known allocation run ID to avoid redundant fetches
    _lastKnownRunId: null,
    _lastPollTime: 0,
    _lastRunCount: 0, // Track number of runs to detect new allocations
    
    // Selected date for dashboard (initialized to today on first access)
    _selectedDate: null,
    
    // Get or initialize the selected date
    getSelectedDate() {
        if (!this._selectedDate) {
            this._selectedDate = this.getTodayDate();
        }
        return this._selectedDate;
    },

    // Event callbacks
    _callbacks: {
        onStatusUpdate: null,
        onWorkflowUpdate: null,
        onError: null
    },

    /**
     * Set the base URL for API calls
     */
    setBaseUrl(url) {
        this.BASE_URL = url;
    },

    /**
     * Make an API request
     */
    async request(endpoint, options = {}) {
        const url = `${this.BASE_URL}${this.API_PREFIX}${endpoint}`;

        // Add cache busting for GET requests to ensure real-time data
        const cacheBuster = options.method === 'GET' || !options.method
            ? (endpoint.includes('?') ? '&' : '?') + `_t=${new Date().getTime()}`
            : '';

        try {
            const response = await fetch(url + cacheBuster, {
                cache: 'no-store',
                headers: {
                    'Content-Type': 'application/json',
                    'Pragma': 'no-cache',
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            if (this._callbacks.onError) {
                this._callbacks.onError(error);
            }
            throw error;
        }
    },

    /**
     * Check system health
     */
    async checkHealth() {
        try {
            const response = await fetch(`${this.BASE_URL}/health`);
            return await response.json();
        } catch (error) {
            return { status: 'disconnected', error: error.message };
        }
    },

    /**
     * Get admin system health with more details
     */
    async getAdminHealth() {
        try {
            return await this.request('/admin/health');
        } catch (error) {
            return null;
        }
    },

    /**
     * Get allocation runs for a specific date
     */
    async getAllocationRuns(date) {
        return this.request(`/admin/allocation_runs?date=${date}`);
    },

    /**
     * Get agent timeline for an allocation run
     */
    async getAgentTimeline(allocationRunId) {
        return this.request(`/admin/agent_timeline?allocation_run_id=${allocationRunId}`);
    },

    /**
     * Get fairness metrics
     */
    async getFairnessMetrics(startDate, endDate) {
        return this.request(`/admin/metrics/fairness?start_date=${startDate}&end_date=${endDate}`);
    },

    /**
     * Trigger a new allocation (for testing)
     */
    async triggerAllocation(data) {
        return this.request('/allocate', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    /**
     * Get driver details
     */
    async getDriver(driverId) {
        return this.request(`/drivers/${driverId}`);
    },

    /**
     * Get route details
     */
    async getRoute(routeId) {
        return this.request(`/routes/${routeId}`);
    },

    /**
     * Get assignments for a date (needed for map visualization)
     */
    async getAssignments(date) {
        return this.request(`/admin/assignments?date=${date}`);
    },

    /**
     * Register callback for status updates
     */
    onStatusUpdate(callback) {
        this._callbacks.onStatusUpdate = callback;
    },

    /**
     * Register callback for workflow updates
     */
    onWorkflowUpdate(callback) {
        this._callbacks.onWorkflowUpdate = callback;
    },

    /**
     * Register callback for errors
     */
    onError(callback) {
        this._callbacks.onError = callback;
    },

    /**
     * Start polling for updates
     */
    startPolling() {
        if (this._pollTimer) {
            clearInterval(this._pollTimer);
        }

        // Use slower polling interval by default
        this._pollTimer = setInterval(async () => {
            await this._pollStatus();
        }, this.POLL_INTERVAL);

        // Initial poll
        this._pollStatus();
    },

    /**
     * Stop polling
     */
    stopPolling() {
        if (this._pollTimer) {
            clearInterval(this._pollTimer);
            this._pollTimer = null;
        }
    },
    
    /**
     * Set the selected date for dashboard queries
     */
    setSelectedDate(date) {
        this._selectedDate = date;
        // Reset cache when date changes
        this._lastKnownRunId = null;
        this._lastRunCount = 0;
    },

    /**
     * Internal polling function - fetches real data from backend
     * Optimized to minimize redundant API calls
     */
    async _pollStatus() {
        try {
            const health = await this.checkHealth();
            const connected = health.status === 'healthy';

            if (this._callbacks.onStatusUpdate) {
                this._callbacks.onStatusUpdate({
                    connected,
                    health
                });
            }

            // If connected, check for new data (but minimize API calls)
            if (connected && this._callbacks.onWorkflowUpdate) {
                const queryDate = this.getSelectedDate();
                
                // Quick check: just get allocation runs count/latest ID
                const runsResponse = await this.getAllocationRuns(queryDate);
                const runs = runsResponse.runs || [];
                const runCount = runs.length;
                const latestRunId = runs.length > 0 ? runs[runs.length - 1].id : null;
                
                // Only fetch full data if:
                // 1. We have no cached data, OR
                // 2. Run count changed (new allocation added), OR  
                // 3. Latest run ID changed
                const needsFullFetch = !this._currentAllocationRun || 
                    runCount !== this._lastRunCount ||
                    latestRunId !== this._lastKnownRunId;
                
                if (needsFullFetch && runs.length > 0) {
                    console.log(`[Poll] New data detected, fetching full workflow state...`);
                    const workflowState = await this.getRealWorkflowState(queryDate);
                    this._lastRunCount = runCount;
                    this._callbacks.onWorkflowUpdate(workflowState);
                }
                // Otherwise, no API call needed - data hasn't changed
            }
        } catch (error) {
            if (this._callbacks.onStatusUpdate) {
                this._callbacks.onStatusUpdate({
                    connected: false,
                    error: error.message
                });
            }
        }
    },

    /**
     * Get today's date in local timezone (YYYY-MM-DD format)
     */
    getTodayDate() {
        const now = new Date();
        const year = now.getFullYear();
        const month = String(now.getMonth() + 1).padStart(2, '0');
        const day = String(now.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    },

    /**
     * Fetch real workflow state from backend
     * @param {string} date - Optional date to fetch, defaults to selected date or today
     * @param {string} runId - Optional specific run ID to fetch
     */
    async getRealWorkflowState(date = null, runId = null) {
        try {
            // If a specific run_id is provided, fetch that run directly
            if (runId) {
                return await this.getWorkflowStateForRun(runId);
            }

            // Use provided date, or get the selected date
            const queryDate = date || this.getSelectedDate();
            const runsResponse = await this.getAllocationRuns(queryDate);

            if (runsResponse.runs && runsResponse.runs.length > 0) {
                // Get the latest allocation run
                const latestRun = runsResponse.runs[runsResponse.runs.length - 1];
                this._currentAllocationRun = latestRun;
                
                // Track the run ID to avoid redundant fetches
                this._lastKnownRunId = latestRun.id;

                // Get agent timeline for this run
                const timeline = await this.getAgentTimeline(latestRun.id);
                this._lastAgentTimeline = timeline;

                // Transform timeline into workflow state
                return this.transformTimelineToWorkflowState(timeline, latestRun);
            }
        } catch (error) {
            console.log('Using mock data - no allocations found:', error.message);
        }

        // Fallback to mock state
        return this.getMockWorkflowState();
    },

    /**
     * Fetch workflow state for a specific run ID
     * @param {string} runId - The allocation run ID to fetch
     */
    async getWorkflowStateForRun(runId) {
        try {
            // Fetch run summary to get run metadata
            const summaryResponse = await fetch(`${this.BASE_URL}${this.API_PREFIX}/runs/${runId}/summary`);
            if (!summaryResponse.ok) {
                throw new Error(`Failed to fetch run summary: ${summaryResponse.status}`);
            }
            const summary = await summaryResponse.json();

            // Build allocation run object from summary
            const allocationRun = {
                id: summary.allocation_run_id,
                date: summary.date,
                num_drivers: summary.num_drivers,
                num_routes: summary.num_routes,
                num_packages: summary.num_packages,
                gini_index: summary.global_gini_index,
                std_dev: summary.global_std_dev,
                status: summary.status,
            };
            this._currentAllocationRun = allocationRun;
            this._lastKnownRunId = runId;

            // Get agent timeline for this specific run
            const timeline = await this.getAgentTimeline(runId);
            this._lastAgentTimeline = timeline;

            // Transform timeline into workflow state
            return this.transformTimelineToWorkflowState(timeline, allocationRun);
        } catch (error) {
            console.error('Error fetching workflow state for run:', runId, error);
            throw error;
        }
    },

    /**
     * Transform agent timeline data to workflow state format
     */
    transformTimelineToWorkflowState(timeline, allocationRun) {
        // Map backend agent names to frontend node IDs
        // Support both UPPERCASE (from LangGraph) and PascalCase formats
        const agentNameMap = {
            // LangGraph format (UPPERCASE)
            'ML_EFFORT': { id: 'route_planner', name: 'Route Planner Agent', type: 'agent' },
            'ML_EFFORT_AGENT': { id: 'route_planner', name: 'Route Planner Agent', type: 'agent' },
            'ROUTE_PLANNER': { id: 'route_planner', name: 'Route Planner Agent', type: 'agent' },
            'FAIRNESS_MANAGER': { id: 'fairness_manager', name: 'Fairness Manager Agent', type: 'agent' },
            'DRIVER_LIAISON': { id: 'driver_liaison', name: 'Driver Liaison Agent', type: 'agent' },
            'EXPLAINABILITY': { id: 'explainability', name: 'Explainability Agent', type: 'agent' },
            'LEARNING': { id: 'learning', name: 'Learning Agent', type: 'agent' },
            // PascalCase format (legacy)
            'MLEffortAgent': { id: 'route_planner', name: 'Route Planner Agent', type: 'agent' },
            'RoutePlannerAgent': { id: 'route_planner', name: 'Route Planner Agent', type: 'agent' },
            'FairnessManagerAgent': { id: 'fairness_manager', name: 'Fairness Manager Agent', type: 'agent' },
            'DriverLiaisonAgent': { id: 'driver_liaison', name: 'Driver Liaison Agent', type: 'agent' },
            'ExplainabilityAgent': { id: 'explainability', name: 'Explainability Agent', type: 'agent' },
            'LearningAgent': { id: 'learning', name: 'Learning Agent', type: 'agent' },
        };


        // Build agent status map from timeline
        // Handle both new format (timeline.timeline) and legacy format (timeline.steps)
        const agentStatus = {};
        const agentLogs = {};
        let totalSteps = 0;

        // Get the actual steps array from the response
        const steps = timeline.timeline || timeline.steps || [];

        if (steps && steps.length > 0) {
            steps.forEach(step => {
                const mapped = agentNameMap[step.agent_name];
                if (mapped) {
                    agentStatus[mapped.id] = 'active';
                    if (!agentLogs[mapped.id]) {
                        agentLogs[mapped.id] = [];
                    }
                    agentLogs[mapped.id].push(step);
                    totalSteps++;
                }
            });
        }

        // Store logs for terminal display
        this._agentLogs = agentLogs;

        const agents = [
            {
                id: 'central_orchestrator',
                name: 'Central Orchestrator Agent',
                description: 'Central Intelligence Hub',
                status: 'active',
                type: 'orchestrator',
                meta: `${Object.keys(agentStatus).length} Active`
            },
            {
                id: 'route_database',
                name: 'Route Database',
                description: `Run: ${allocationRun.id.substring(0, 8)}...`,
                status: 'active',
                type: 'database',
                meta: 'Connected'
            },
            {
                id: 'route_planner',
                name: 'Route Planner Agent',
                description: agentLogs['route_planner'] ?
                    `${agentLogs['route_planner'].length} decisions` : 'Optimizing routes',
                status: agentStatus['route_planner'] || 'idle',
                type: 'agent',
                meta: null
            },
            {
                id: 'fairness_manager',
                name: 'Fairness Manager Agent',
                description: agentLogs['fairness_manager'] ?
                    `Gini: ${allocationRun.gini_index?.toFixed(3) || 'N/A'}` : 'Ensuring equitable distribution',
                status: agentStatus['fairness_manager'] || 'idle',
                type: 'agent',
                meta: null
            },
            {
                id: 'driver_liaison',
                name: 'Driver Liaison Agent',
                description: agentLogs['driver_liaison'] ?
                    `${agentLogs['driver_liaison'].length} interactions` : 'Driver communication hub',
                status: agentStatus['driver_liaison'] || 'idle',
                type: 'agent',
                meta: null
            },
            {
                id: 'explainability',
                name: 'Explainability Agent',
                description: agentLogs['explainability'] ?
                    `${agentLogs['explainability'].length} explanations` : 'Generating explanations',
                status: agentStatus['explainability'] || 'idle',
                type: 'agent',
                meta: null
            },
            {
                id: 'learning',
                name: 'Learning Agent',
                description: 'Continuous improvement loop',
                status: agentStatus['learning'] || 'idle',
                type: 'agent',
                meta: null
            }
        ];

        // Build connections - active if agent has data
        const connections = [
            { from: 'central_orchestrator', to: 'route_database', active: true },
            { from: 'central_orchestrator', to: 'route_planner', active: !!agentStatus['route_planner'] },
            { from: 'central_orchestrator', to: 'fairness_manager', active: !!agentStatus['fairness_manager'] },
            { from: 'central_orchestrator', to: 'driver_liaison', active: !!agentStatus['driver_liaison'] },
            { from: 'central_orchestrator', to: 'explainability', active: !!agentStatus['explainability'] },
            { from: 'central_orchestrator', to: 'learning', active: !!agentStatus['learning'] },
            { from: 'route_database', to: 'route_planner', active: !!agentStatus['route_planner'] },
            { from: 'route_planner', to: 'fairness_manager', active: !!agentStatus['fairness_manager'] },
            { from: 'fairness_manager', to: 'driver_liaison', active: !!agentStatus['driver_liaison'] },
            { from: 'driver_liaison', to: 'explainability', active: !!agentStatus['explainability'] },
            { from: 'explainability', to: 'learning', active: !!agentStatus['learning'] }
        ];

        const activeConnections = connections.filter(c => c.active).length;

        return {
            agents,
            connections,
            stats: {
                processing: 0,
                dataFlows: activeConnections,
                totalAgents: Object.keys(agentStatus).length
            },
            allocationRun,
            timeline,
            isRealData: true
        };
    },

    /**
     * Get cached agent logs for terminal display
     */
    getAgentLogs(agentId) {
        if (this._agentLogs && this._agentLogs[agentId]) {
            return this._agentLogs[agentId];
        }
        return null;
    },

    /**
     * Get current allocation run
     */
    getCurrentAllocationRun() {
        return this._currentAllocationRun;
    },

    /**
     * Get mock workflow state for demo mode
     */
    getMockWorkflowState() {
        return {
            agents: [
                {
                    id: 'central_orchestrator',
                    name: 'Central Orchestrator Agent',
                    description: 'Central Intelligence Hub',
                    status: 'active',
                    type: 'orchestrator',
                    meta: '6 Agents'
                },
                {
                    id: 'route_database',
                    name: 'Route Database',
                    description: '@ 18.9K records',
                    status: 'active',
                    type: 'database',
                    meta: 'Connected'
                },
                {
                    id: 'route_planner',
                    name: 'Route Planner Agent',
                    description: 'Optimizing delivery routes',
                    status: 'processing',
                    type: 'agent',
                    meta: null
                },
                {
                    id: 'fairness_manager',
                    name: 'Fairness Manager Agent',
                    description: 'Ensuring equitable distribution',
                    status: 'idle',
                    type: 'agent',
                    meta: null
                },
                {
                    id: 'driver_liaison',
                    name: 'Driver Liaison Agent',
                    description: 'Driver communication hub',
                    status: 'idle',
                    type: 'agent',
                    meta: null
                },
                {
                    id: 'explainability',
                    name: 'Explainability Agent',
                    description: 'Generating detailed explanations',
                    status: 'idle',
                    type: 'agent',
                    meta: null
                },
                {
                    id: 'learning',
                    name: 'Learning Agent',
                    description: 'Continuous improvement loop',
                    status: 'idle',
                    type: 'agent',
                    meta: null
                }
            ],
            connections: [
                { from: 'central_orchestrator', to: 'route_database', active: true },
                { from: 'central_orchestrator', to: 'route_planner', active: true },
                { from: 'central_orchestrator', to: 'fairness_manager', active: true },
                { from: 'central_orchestrator', to: 'driver_liaison', active: true },
                { from: 'central_orchestrator', to: 'explainability', active: true },
                { from: 'central_orchestrator', to: 'learning', active: true },
                { from: 'route_database', to: 'route_planner', active: false },
                { from: 'route_planner', to: 'fairness_manager', active: false },
                { from: 'fairness_manager', to: 'driver_liaison', active: false },
                { from: 'driver_liaison', to: 'explainability', active: false },
                { from: 'explainability', to: 'learning', active: false }
            ],
            stats: {
                processing: 1,
                dataFlows: 6,
                totalAgents: 6
            },
            isRealData: false
        };
    }
};

// Export for use in app.js
window.API = API;

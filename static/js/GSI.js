// Utility functions
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const formatTime = (timestamp) => new Date(timestamp).toLocaleTimeString('en-US', { 
    hour12: false, 
    timeZone: 'UTC',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
  const timeAgo = (timestamp) => {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  // Advanced threat categories for professional use
  const THREAT_CATEGORIES = {
    kinetic: { color: '#DC2626', icon: 'fas fa-explosion', name: 'Kinetic Threat' },
    cyber: { color: '#8B5CF6', icon: 'fas fa-laptop-code', name: 'Cyber Threat' },
    space: { color: '#3B82F6', icon: 'fas fa-satellite', name: 'Space Debris' },
    environmental: { color: '#10B981', icon: 'fas fa-leaf', name: 'Environmental' },
    infrastructure: { color: '#F59E0B', icon: 'fas fa-building', name: 'Critical Infrastructure' },
    wmd: { color: '#B91C1C', icon: 'fas fa-radiation', name: 'WMD Threat' }
  };

  const SEVERITY_LEVELS = {
    1: { name: 'INFO', color: '#6B7280', priority: 'Informational' },
    2: { name: 'LOW', color: '#059669', priority: 'Low Priority' },
    3: { name: 'MEDIUM', color: '#D97706', priority: 'Medium Priority' },
    4: { name: 'HIGH', color: '#DC2626', priority: 'High Priority' },
    5: { name: 'CRITICAL', color: '#B91C1C', priority: 'Critical Priority' }
  };

  // Data Store
  class ThreatDataStore {
    constructor() {
      this.threats = [];
      this.subscribers = [];
    }

    addThreat(threat) {
      threat.id = crypto.randomUUID();
      threat.timestamp = Date.now();
      this.threats.unshift(threat);
      this.notify('add', threat);
    }

    updateThreat(id, updates) {
      const index = this.threats.findIndex(t => t.id === id);
      if (index !== -1) {
        this.threats[index] = { ...this.threats[index], ...updates };
        this.notify('update', this.threats[index]);
      }
    }

    removeThreat(id) {
      const index = this.threats.findIndex(t => t.id === id);
      if (index !== -1) {
        const threat = this.threats.splice(index, 1)[0];
        this.notify('remove', threat);
      }
    }

    getThreats(filters = {}) {
      return this.threats.filter(threat => {
        if (filters.category && filters.category !== 'all' && threat.category !== filters.category) return false;
        if (filters.severity && threat.severity < filters.severity) return false;
        if (filters.timeWindow) {
          const cutoff = Date.now() - (filters.timeWindow * 60 * 60 * 1000);
          if (threat.timestamp < cutoff) return false;
        }
        if (filters.source && filters.source !== 'all' && threat.source !== filters.source) return false;
        if (filters.search) {
          const searchLower = filters.search.toLowerCase();
          const searchableText = `${threat.title} ${threat.description} ${threat.location?.country || ''} ${threat.location?.region || ''}`.toLowerCase();
          if (!searchableText.includes(searchLower)) return false;
        }
        return true;
      });
    }

    subscribe(callback) {
      this.subscribers.push(callback);
      return () => {
        const index = this.subscribers.indexOf(callback);
        if (index > -1) this.subscribers.splice(index, 1);
      };
    }

    notify(action, threat) {
      this.subscribers.forEach(callback => callback(action, threat));
    }

    clear() {
      this.threats = [];
      this.notify('clear');
    }
  }

  // Map Controller
  class AdvancedMapController {
    constructor(elementId) {
      this.map = L.map(elementId, {
        center: [20, 0],
        zoom: 3,
        minZoom: 2,
        maxZoom: 18,
        worldCopyJump: true,
        zoomControl: false
      });

      // Dark satellite-style base layer
      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        
      }).addTo(this.map);

      // Overlay for labels
      this.labelsLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; CartoDB',
        pane: 'shadowPane'
      }).addTo(this.map);

      this.threatMarkers = new Map();
      this.clusterGroup = L.markerClusterGroup({
        chunkedLoading: true,
        maxClusterRadius: 50,
        iconCreateFunction: (cluster) => {
          const count = cluster.getChildCount();
          const severity = this.getClusterSeverity(cluster);
          return L.divIcon({
            html: `<div class="cluster-marker severity-${severity}">${count}</div>`,
            className: 'custom-cluster-icon',
            iconSize: L.point(40, 40)
          });
        }
      });

      this.heatmapLayer = L.heatLayer([], {
        radius: 30,
        blur: 20,
        maxZoom: 10,
        gradient: {
          0.2: '#059669',
          0.4: '#D97706', 
          0.6: '#DC2626',
          0.8: '#B91C1C',
          1.0: '#7C2D12'
        }
      });

      this.useCluster = true;
      this.useHeatmap = false;
      this.threatMarkers = new Map();

      this.initializeMapControls();

      // Ensure map resizes correctly
      setTimeout(() => {
        this.map.invalidateSize();
      }, 100);

      // Handle window resize
      window.addEventListener('resize', () => {
        this.map.invalidateSize();
      });
    }

    getClusterSeverity(cluster) {
      let maxSeverity = 1;
      cluster.getAllChildMarkers().forEach(marker => {
        if (marker.options.threat && marker.options.threat.severity > maxSeverity) {
          maxSeverity = marker.options.threat.severity;
        }
      });
      return maxSeverity;
    }

    createThreatMarker(threat) {
      const category = THREAT_CATEGORIES[threat.category] || THREAT_CATEGORIES.kinetic;
      const severity = SEVERITY_LEVELS[threat.severity] || SEVERITY_LEVELS[1];
      
      const icon = L.divIcon({
        html: `
          <div class="threat-marker severity-${threat.severity}" style="background-color: ${category.color}">
            <i class="${category.icon}"></i>
            ${threat.severity >= 4 ? '<div class="pulse-ring"></div>' : ''}
          </div>
        `,
        className: 'custom-threat-marker',
        iconSize: [24, 24],
        iconAnchor: [12, 12]
      });

      const marker = L.marker([threat.location.lat, threat.location.lng], {
        icon,
        threat
      });

      marker.bindPopup(`
        <div class="threat-popup">
          <div class="popup-header">
            <div class="threat-category">
              <i class="${category.icon}"></i>
              ${category.name}
            </div>
            <div class="severity-badge severity-${threat.severity}">
              ${severity.name}
            </div>
          </div>
          <h3>${threat.title}</h3>
          <p>${threat.description}</p>
          <div class="popup-meta">
            <div class="meta-item">
              <i class="fas fa-map-marker-alt"></i>
              ${threat.location.country}, ${threat.location.region}
            </div>
            <div class="meta-item">
              <i class="fas fa-clock"></i>
              ${formatTime(threat.timestamp)} UTC
            </div>
            <div class="meta-item">
              <i class="fas fa-database"></i>
              Source: ${threat.source}
            </div>
            <div class="meta-item">
              <i class="fas fa-percent"></i>
              Confidence: ${threat.confidence}%
            </div>
          </div>
          <div class="popup-actions">
            <button class="btn-details" data-threat-id="${threat.id}">
              <i class="fas fa-info-circle"></i> Details
            </button>
            <button class="btn-track" data-threat-id="${threat.id}">
              <i class="fas fa-crosshairs"></i> Track
            </button>
          </div>
        </div>
      `, { maxWidth: 350 });

      return marker;
    }

    addThreat(threat) {
      if (this.threatMarkers.has(threat.id)) return;
      
      const marker = this.createThreatMarker(threat);
      this.threatMarkers.set(threat.id, marker);
      
      if (this.useCluster) {
        this.clusterGroup.addLayer(marker);
      } else {
        marker.addTo(this.map);
      }
      
      this.updateHeatmap();
    }

    updateThreat(threat) {
      if (this.threatMarkers.has(threat.id)) {
        this.removeThreat(threat.id);
        this.addThreat(threat);
      }
    }

    removeThreat(threatId) {
      const marker = this.threatMarkers.get(threatId);
      if (marker) {
        if (this.useCluster) {
          this.clusterGroup.removeLayer(marker);
        } else {
          this.map.removeLayer(marker);
        }
        this.threatMarkers.delete(threatId);
        this.updateHeatmap();
      }
    }

    clearThreats() {
      this.threatMarkers.forEach(marker => {
        if (this.useCluster) {
          this.clusterGroup.removeLayer(marker);
        } else {
          this.map.removeLayer(marker);
        }
      });
      this.threatMarkers.clear();
      this.updateHeatmap();
    }

    toggleCluster(enabled) {
      this.useCluster = enabled;
      
      if (enabled) {
        this.threatMarkers.forEach(marker => {
          this.map.removeLayer(marker);
          this.clusterGroup.addLayer(marker);
        });
        this.map.addLayer(this.clusterGroup);
      } else {
        this.map.removeLayer(this.clusterGroup);
        this.threatMarkers.forEach(marker => {
          this.clusterGroup.removeLayer(marker);
          marker.addTo(this.map);
        });
      }
    }

    toggleHeatmap(enabled) {
      this.useHeatmap = enabled;
      
      if (enabled) {
        this.map.addLayer(this.heatmapLayer);
        this.updateHeatmap();
      } else {
        this.map.removeLayer(this.heatmapLayer);
      }
    }

    updateHeatmap() {
      if (!this.useHeatmap) return;
      
      const heatPoints = Array.from(this.threatMarkers.values()).map(marker => {
        const threat = marker.options.threat;
        return [
          threat.location.lat,
          threat.location.lng,
          threat.severity / 5 // Normalize severity to 0-1 range
        ];
      });
      
      this.heatmapLayer.setLatLngs(heatPoints);
    }

    initializeMapControls() {
      // Custom CSS for map elements
      const style = document.createElement('style');
      style.textContent = `
        .custom-threat-marker {
          background: none;
          border: none;
        }
        
        .threat-marker {
          width: 24px;
          height: 24px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          position: relative;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
          border: 2px solid rgba(255, 255, 255, 0.3);
        }
        
        .threat-marker i {
          color: white;
          font-size: 10px;
          text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8);
        }
        
        .pulse-ring {
          position: absolute;
          top: -4px;
          left: -4px;
          width: 32px;
          height: 32px;
          border: 2px solid currentColor;
          border-radius: 50%;
          opacity: 0.6;
          animation: pulse-ring 2s ease-out infinite;
        }
        
        @keyframes pulse-ring {
          0% { transform: scale(0.8); opacity: 1; }
          100% { transform: scale(2); opacity: 0; }
        }
        
        .cluster-marker {
          background: rgba(59, 130, 246, 0.9);
          color: white;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: bold;
          font-size: 12px;
          border: 2px solid rgba(255, 255, 255, 0.3);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        }
        
        .cluster-marker.severity-5 { background: rgba(185, 28, 28, 0.9); }
        .cluster-marker.severity-4 { background: rgba(220, 38, 38, 0.9); }
        .cluster-marker.severity-3 { background: rgba(217, 119, 6, 0.9); }
        
        .threat-popup {
          font-family: 'Inter', sans-serif;
          color: #F8FAFC;
        }
        
        .popup-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
          padding-bottom: 8px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .threat-category {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 12px;
          font-weight: 500;
          color: #CBD5E1;
        }
        
        .severity-badge {
          padding: 2px 8px;
          border-radius: 12px;
          font-size: 10px;
          font-weight: bold;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .severity-badge.severity-5 { background: #B91C1C; color: white; }
        .severity-badge.severity-4 { background: #DC2626; color: white; }
        .severity-badge.severity-3 { background: #D97706; color: white; }
        .severity-badge.severity-2 { background: #059669; color: white; }
        .severity-badge.severity-1 { background: #6B7280; color: white; }
        
        .threat-popup h3 {
          margin: 0 0 8px 0;
          font-size: 16px;
          font-weight: 600;
          color: white;
        }
        
        .threat-popup p {
          margin: 0 0 12px 0;
          font-size: 13px;
          line-height: 1.5;
          color: #CBD5E1;
        }
        
        .popup-meta {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
          margin-bottom: 12px;
        }
        
        .meta-item {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 11px;
          color: #94A3B8;
        }
        
        .meta-item i {
          width: 12px;
          color: #64748B;
        }
        
        .popup-actions {
          display: flex;
          gap: 8px;
        }
        
        .popup-actions button {
          flex: 1;
          padding: 6px 12px;
          border: none;
          border-radius: 6px;
          font-size: 11px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s ease;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 4px;
        }
        
        .btn-details {
          background: rgba(59, 130, 246, 0.2);
          color: #60A5FA;
          border: 1px solid rgba(59, 130, 246, 0.3);
        }
        
        .btn-details:hover {
          background: rgba(59, 130, 246, 0.3);
        }
        
        .btn-track {
          background: rgba(34, 197, 94, 0.2);
          color: #4ADE80;
          border: 1px solid rgba(34, 197, 94, 0.3);
        }
        
        .btn-track:hover {
          background: rgba(34, 197, 94, 0.3);
        }
      `;
      document.head.appendChild(style);
    }

    focusOnThreat(threatId) {
      const marker = this.threatMarkers.get(threatId);
      if (marker) {
        this.map.setView(marker.getLatLng(), 10);
        marker.openPopup();
      }
    }
  }

  // Analytics Controller
  class AnalyticsController {
    constructor() {
      this.initializeCharts();
    }

    initializeCharts() {
      // Timeline Chart
      this.timelineChart = new Chart($('#timelineChart'), {
        type: 'line',
        data: {
          labels: [],
          datasets: [{
            label: 'Threat Activity',
            data: [],
            borderColor: '#DC2626',
            backgroundColor: 'rgba(220, 38, 38, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 2,
            pointHoverRadius: 4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: 'rgba(15, 20, 25, 0.95)',
              titleColor: '#F8FAFC',
              bodyColor: '#CBD5E1',
              borderColor: 'rgba(255, 255, 255, 0.1)',
              borderWidth: 1
            }
          },
          scales: {
            x: {
              type: 'time',
              time: { unit: 'hour' },
              grid: { color: 'rgba(255, 255, 255, 0.1)' },
              ticks: { color: '#64748B', font: { family: 'JetBrains Mono' } }
            },
            y: {
              beginAtZero: true,
              grid: { color: 'rgba(255, 255, 255, 0.1)' },
              ticks: { color: '#64748B', font: { family: 'JetBrains Mono' } }
            }
          }
        }
      });

      // Threat Matrix Chart
      this.threatMatrixChart = new Chart($('#threatMatrix'), {
        type: 'doughnut',
        data: {
          labels: Object.keys(THREAT_CATEGORIES).map(key => THREAT_CATEGORIES[key].name),
          datasets: [{
            data: [],
            backgroundColor: Object.keys(THREAT_CATEGORIES).map(key => THREAT_CATEGORIES[key].color),
            borderWidth: 0
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'bottom',
              labels: {
                color: '#CBD5E1',
                font: { family: 'Inter', size: 11 },
                padding: 15,
                usePointStyle: true
              }
            },
            tooltip: {
              backgroundColor: 'rgba(15, 20, 25, 0.95)',
              titleColor: '#F8FAFC',
              bodyColor: '#CBD5E1',
              borderColor: 'rgba(255, 255, 255, 0.1)',
              borderWidth: 1
            }
          },
          cutout: '60%'
        }
      });
    }

    updateCharts(threats) {
      this.updateTimelineChart(threats);
      this.updateThreatMatrixChart(threats);
    }

    updateTimelineChart(threats) {
      const now = Date.now();
      const hours = 24;
      const timeSlots = [];
      
      for (let i = hours - 1; i >= 0; i--) {
        timeSlots.push(new Date(now - (i * 60 * 60 * 1000)));
      }
      
      const counts = timeSlots.map(slot => {
        const slotStart = slot.getTime();
        const slotEnd = slotStart + (60 * 60 * 1000);
        return threats.filter(t => t.timestamp >= slotStart && t.timestamp < slotEnd).length;
      });

      this.timelineChart.data.labels = timeSlots;
      this.timelineChart.data.datasets[0].data = counts;
      this.timelineChart.update('none');
    }

    updateThreatMatrixChart(threats) {
      const categoryCounts = {};
      Object.keys(THREAT_CATEGORIES).forEach(key => {
        categoryCounts[key] = threats.filter(t => t.category === key).length;
      });

      this.threatMatrixChart.data.datasets[0].data = Object.values(categoryCounts);
      this.threatMatrixChart.update('none');
    }
  }

  // Notification System
  class NotificationSystem {
    constructor() {
      this.container = $('#toastContainer');
    }

    show(message, type = 'info', duration = 5000) {
      const toast = document.createElement('div');
      toast.className = `glass-panel rounded-lg p-4 border-l-4 transition-all duration-300 transform translate-x-full opacity-0`;
      
      const colors = {
        info: 'border-blue-500 bg-blue-500/10',
        success: 'border-green-500 bg-green-500/10',
        warning: 'border-amber-500 bg-amber-500/10',
        error: 'border-red-500 bg-red-500/10',
        critical: 'border-red-700 bg-red-700/20'
      };

      const icons = {
        info: 'fas fa-info-circle text-blue-400',
        success: 'fas fa-check-circle text-green-400',
        warning: 'fas fa-exclamation-triangle text-amber-400',
        error: 'fas fa-exclamation-circle text-red-400',
        critical: 'fas fa-radiation text-red-400'
      };

      toast.classList.add(...colors[type].split(' '));
      toast.innerHTML = `
        <div class="flex items-start gap-3">
          <i class="${icons[type]}"></i>
          <div class="flex-1">
            <div class="text-sm font-medium text-white terminal-font">${message.title || 'Notification'}</div>
            ${message.body ? `<div class="text-xs text-slate-300 mt-1">${message.body}</div>` : ''}
          </div>
          <button class="text-slate-400 hover:text-white transition-colors" onclick="this.parentElement.parentElement.remove()">
            <i class="fas fa-times"></i>
          </button>
        </div>
      `;

      this.container.appendChild(toast);

      // Animate in
      requestAnimationFrame(() => {
        toast.classList.remove('translate-x-full', 'opacity-0');
      });

      // Auto remove
      setTimeout(() => {
        toast.classList.add('translate-x-full', 'opacity-0');
        setTimeout(() => toast.remove(), 300);
      }, duration);

      return toast;
    }
  }

  // Main Application
  class GlobalSecurityIntelligence {
    constructor() {
      this.dataStore = new ThreatDataStore();
      this.mapController = new AdvancedMapController('map');
      this.analyticsController = new AnalyticsController();
      this.notificationSystem = new NotificationSystem();
      
      this.filters = {
        category: 'all',
        severity: 1,
        timeWindow: 24,
        source: 'all',
        search: ''
      };

      this.initialize();
    }

    initialize() {
      this.setupEventListeners();
      this.startRealTimeUpdates();
      this.generateSampleData();
      this.updateDashboard();
      this.startTimeUpdate();
    }

    setupEventListeners() {
      // Filter controls
      $('#threatFilter').addEventListener('change', (e) => {
        this.filters.category = e.target.value;
        this.updateDashboard();
      });

      $('#severityFilter').addEventListener('change', (e) => {
        this.filters.severity = parseInt(e.target.value);
        this.updateDashboard();
      });

      $('#timeWindow').addEventListener('change', (e) => {
        this.filters.timeWindow = parseInt(e.target.value);
        this.updateDashboard();
      });

      $('#sourceFilter').addEventListener('change', (e) => {
        this.filters.source = e.target.value;
        this.updateDashboard();
      });

      $('#searchInput').addEventListener('input', (e) => {
        this.filters.search = e.target.value;
        this.updateDashboard();
      });

      // Map controls
      $('#clusterToggle').addEventListener('click', (e) => {
        const enabled = e.target.classList.contains('bg-blue-600');
        this.mapController.toggleCluster(!enabled);
        e.target.classList.toggle('bg-blue-600');
        e.target.classList.toggle('bg-space-600');
      });

      $('#heatToggle').addEventListener('click', (e) => {
        const enabled = e.target.classList.contains('bg-amber-600');
        this.mapController.toggleHeatmap(!enabled);
        e.target.classList.toggle('bg-amber-600');
        e.target.classList.toggle('bg-space-600');
      });

      // Map navigation
      $('#zoomInBtn').addEventListener('click', () => this.mapController.map.zoomIn());
      $('#zoomOutBtn').addEventListener('click', () => this.mapController.map.zoomOut());

      $('#satelliteLayerBtn').addEventListener('click', () => {
        if (this.mapController.labelsLayer.isAdded) {
          this.mapController.map.removeLayer(this.mapController.labelsLayer);
          this.mapController.labelsLayer.isAdded = false;
        } else {
          this.mapController.map.addLayer(this.mapController.labelsLayer);
          this.mapController.labelsLayer.isAdded = true;
        }
      });

      $('#weatherLayerBtn').addEventListener('click', () => {
        this.notificationSystem.show({
          title: 'Weather Layer',
          body: 'Weather overlay toggle not implemented yet.'
        }, 'info');
      });

      $('#locateBtn').addEventListener('click', () => {
        this.mapController.map.setView([20, 0], 3);
      });

      // Sidebar controls
      $('#sidebarToggle').addEventListener('click', () => {
        $('.sidebar').classList.toggle('open');
        setTimeout(() => this.mapController.map.invalidateSize(), 300);
      });

      // Threat matrix controls
      $$('.threat-cell').forEach(cell => {
        cell.addEventListener('click', (e) => {
          const level = e.target.dataset.level;
          this.filters.severity = {
            low: 2,
            medium: 3,
            high: 4,
            critical: 5
          }[level] || 1;
          $('#severityFilter').value = this.filters.severity;
          this.updateDashboard();
        });
      });

      // Intelligence feed controls
      $('#pauseFeedBtn').addEventListener('click', () => {
        this.paused = !this.paused;
        $('#pauseFeedBtn').innerHTML = this.paused 
          ? '<i class="fas fa-play"></i> RESUME'
          : '<i class="fas fa-pause"></i> PAUSE';
      });

      $('#clearFeedBtn').addEventListener('click', () => {
        this.dataStore.clear();
        $('#intelligenceFeed').innerHTML = '';
      });

      // Modal controls
      $('#alertsBtn').addEventListener('click', () => {
        $('#alertModal').classList.add('active');
        this.showAlertDetails();
      });

      $('#closeAlertModal').addEventListener('click', () => {
        $('#alertModal').classList.remove('active');
      });

      // Fullscreen toggle
      $('#fullscreenBtn').addEventListener('click', () => {
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen();
          $('#fullscreenBtn').innerHTML = '<i class="fas fa-compress text-slate-400"></i>';
        } else {
          document.exitFullscreen();
          $('#fullscreenBtn').innerHTML = '<i class="fas fa-expand text-slate-400"></i>';
        }
        setTimeout(() => this.mapController.map.invalidateSize(), 100);
      });

      // Popup actions
      document.addEventListener('click', (e) => {
        if (e.target.classList.contains('btn-details')) {
          const threatId = e.target.dataset.threatId;
          const threat = this.dataStore.threats.find(t => t.id === threatId);
          if (threat) {
            this.showThreatDetails(threat);
          }
        }
        if (e.target.classList.contains('btn-track')) {
          const threatId = e.target.dataset.threatId;
          this.mapController.focusOnThreat(threatId);
        }
      });

      // Data store subscription
      this.dataStore.subscribe((action, threat) => {
        if (action === 'add') {
          this.mapController.addThreat(threat);
          this.addToIntelligenceFeed(threat);
          if (!this.paused && threat.severity >= 4) {
            this.notificationSystem.show({
              title: `${SEVERITY_LEVELS[threat.severity].name} ${THREAT_CATEGORIES[threat.category].name}`,
              body: threat.title
            }, threat.severity >= 5 ? 'critical' : 'error');
          }
        } else if (action === 'update') {
          this.mapController.updateThreat(threat);
        } else if (action === 'remove') {
          this.mapController.removeThreat(threat.id);
        } else if (action === 'clear') {
          this.mapController.clearThreats();
        }
        this.updateDashboard();
      });
    }

    startTimeUpdate() {
      const updateTime = () => {
        $('#utcTime').textContent = formatTime(Date.now());
        $('#lastUpdate').textContent = formatTime(Date.now()) + ' UTC';
      };
      updateTime();
      setInterval(updateTime, 1000);
    }

    generateSampleData() {
      const sampleThreats = [
        {
          category: 'kinetic',
          severity: 5,
          title: 'Missile Launch Detected',
          description: 'Satellite imagery confirms multiple missile launches from military installation.',
          location: { lat: 48.3794, lng: 31.1656, country: 'Ukraine', region: 'Eastern Europe' },
          source: 'satellite',
          confidence: 95
        },
        {
          category: 'cyber',
          severity: 3,
          title: 'DDoS Attack on Infrastructure',
          description: 'Significant distributed denial-of-service attack targeting financial sector.',
          location: { lat: 40.7128, lng: -74.0060, country: 'United States', region: 'North America' },
          source: 'sigint',
          confidence: 88
        },
        {
          category: 'space',
          severity: 2,
          title: 'Debris Field Expansion',
          description: 'New debris field detected in low Earth orbit, potential risk to satellites.',
          location: { lat: 0, lng: 0, country: 'N/A', region: 'LEO' },
          source: 'satellite',
          confidence: 92
        },
        {
          category: 'environmental',
          severity: 4,
          title: 'Wildfire Outbreak',
          description: 'Large-scale wildfires detected via thermal imaging, threatening infrastructure.',
          location: { lat: -33.8688, lng: 151.2093, country: 'Australia', region: 'Oceania' },
          source: 'satellite',
          confidence: 90
        },
        {
          category: 'infrastructure',
          severity: 3,
          title: 'Power Grid Anomaly',
          description: 'Unusual activity detected in regional power grid, possible cyber intrusion.',
          location: { lat: 51.5074, lng: -0.1278, country: 'United Kingdom', region: 'Western Europe' },
          source: 'osint',
          confidence: 85
        }
      ];

      sampleThreats.forEach((threat, index) => {
        setTimeout(() => {
          this.dataStore.addThreat({
            ...threat,
            timestamp: Date.now() - (index * 5 * 60 * 1000)
          });
        }, index * 1000);
      });
    }

    addToIntelligenceFeed(threat) {
      const feed = $('#intelligenceFeed');
      const category = THREAT_CATEGORIES[threat.category] || THREAT_CATEGORIES.kinetic;
      const severity = SEVERITY_LEVELS[threat.severity] || SEVERITY_LEVELS[1];

      const item = document.createElement('div');
      item.className = `glass-panel p-4 rounded-lg border-l-4 border-[${category.color}]`;
      item.innerHTML = `
        <div class="flex items-start gap-3">
          <i class="${category.icon} text-[${category.color}] text-lg"></i>
          <div class="flex-1">
            <div class="flex items-center justify-between">
              <div class="text-sm font-medium text-white">${threat.title}</div>
              <div class="text-xs text-slate-400 terminal-font">${timeAgo(threat.timestamp)}</div>
            </div>
            <div class="text-xs text-slate-300 mt-1">${threat.description}</div>
            <div class="flex items-center gap-4 mt-2 text-xs">
              <div class="flex items-center gap-1">
                <i class="fas fa-map-marker-alt text-slate-400"></i>
                <span>${threat.location.country}, ${threat.location.region}</span>
              </div>
              <div class="flex items-center gap-1">
                <i class="fas fa-database text-slate-400"></i>
                <span>${threat.source}</span>
              </div>
              <div class="severity-badge severity-${threat.severity}">
                ${severity.name}
              </div>
            </div>
          </div>
        </div>
      `;

      feed.insertBefore(item, feed.firstChild);
      if (feed.children.length > 50) {
        feed.lastChild.remove();
      }
    }

    showThreatDetails(threat) {
      const category = THREAT_CATEGORIES[threat.category] || THREAT_CATEGORIES.kinetic;
      const severity = SEVERITY_LEVELS[threat.severity] || SEVERITY_LEVELS[1];

      $('#alertModalContent').innerHTML = `
        <div class="space-y-4">
          <div class="flex items-center gap-3">
            <i class="${category.icon} text-[${category.color}] text-2xl"></i>
            <div>
              <h3 class="text-lg font-semibold text-white">${threat.title}</h3>
              <div class="text-xs text-slate-400 terminal-font">
                ${formatTime(threat.timestamp)} UTC • ${threat.location.country}, ${threat.location.region}
              </div>
            </div>
          </div>
          <div class="border-l-4 border-[${category.color}] pl-4">
            <p class="text-sm text-slate-300">${threat.description}</p>
          </div>
          <div class="grid grid-cols-2 gap-4 text-sm">
            <div>
              <div class="text-xs text-slate-400 terminal-font">Category</div>
              <div class="flex items-center gap-2">
                <i class="${category.icon} text-[${category.color}]"></i>
                <span>${category.name}</span>
              </div>
            </div>
            <div>
              <div class="text-xs text-slate-400 terminal-font">Severity</div>
              <div class="severity-badge severity-${threat.severity}">${severity.name}</div>
            </div>
            <div>
              <div class="text-xs text-slate-400 terminal-font">Source</div>
              <div class="flex items-center gap-2">
                <i class="fas fa-database text-slate-400"></i>
                <span>${threat.source}</span>
              </div>
            </div>
            <div>
              <div class="text-xs text-slate-400 terminal-font">Confidence</div>
              <div class="flex items-center gap-2">
                <i class="fas fa-percent text-slate-400"></i>
                <span>${threat.confidence}%</span>
              </div>
            </div>
          </div>
          <div class="flex gap-4">
            <button class="btn-details bg-blue-500/20 text-blue-300 border border-blue-500/30 px-4 py-2 rounded-lg flex-1" data-threat-id="${threat.id}">
              <i class="fas fa-info-circle"></i> Full Report
            </button>
            <button class="btn-track bg-green-500/20 text-green-300 border border-green-500/30 px-4 py-2 rounded-lg flex-1" data-threat-id="${threat.id}">
              <i class="fas fa-crosshairs"></i> Track on Map
            </button>
          </div>
        </div>
      `;

      $('#alertModal').classList.add('active');
    }

    showAlertDetails() {
      const threats = this.dataStore.getThreats({ severity: 4 });
      if (threats.length === 0) {
        $('#alertModalContent').innerHTML = `
          <div class="text-center text-slate-300">
            <i class="fas fa-bell text-2xl text-slate-400 mb-4"></i>
            <p>No high or critical threats at this time.</p>
          </div>
        `;
        return;
      }

      const content = threats.map(threat => {
        const category = THREAT_CATEGORIES[threat.category] || THREAT_CATEGORIES.kinetic;
        const severity = SEVERITY_LEVELS[threat.severity] || SEVERITY_LEVELS[1];
        return `
          <div class="glass-panel p-4 rounded-lg mb-4 border-l-4 border-[${category.color}]">
            <div class="flex items-start gap-3">
              <i class="${category.icon} text-[${category.color}] text-lg"></i>
              <div class="flex-1">
                <div class="flex items-center justify-between">
                  <div class="text-sm font-medium text-white">${threat.title}</div>
                  <div class="text-xs text-slate-400 terminal-font">${timeAgo(threat.timestamp)}</div>
                </div>
                <div class="text-xs text-slate-300 mt-1">${threat.description}</div>
                <div class="flex items-center gap-4 mt-2 text-xs">
                  <div class="flex items-center gap-1">
                    <i class="fas fa-map-marker-alt text-slate-400"></i>
                    <span>${threat.location.country}, ${threat.location.region}</span>
                  </div>
                  <div class="flex items-center gap-1">
                    <i class="fas fa-database text-slate-400"></i>
                    <span>${threat.source}</span>
                  </div>
                  <div class="severity-badge severity-${threat.severity}">
                    ${severity.name}
                  </div>
                </div>
                <div class="flex gap-2 mt-3">
                  <button class="btn-details bg-blue-500/20 text-blue-300 border border-blue-500/30 px-3 py-1 rounded-lg text-xs" data-threat-id="${threat.id}">
                    <i class="fas fa-info-circle"></i> Details
                  </button>
                  <button class="btn-track bg-green-500/20 text-green-300 border border-green-500/30 px-3 py-1 rounded-lg text-xs" data-threat-id="${threat.id}">
                    <i class="fas fa-crosshairs"></i> Track
                  </button>
                </div>
              </div>
            </div>
          </div>
        `;
      }).join('');

      $('#alertModalContent').innerHTML = content;
    }

    updateDashboard() {
      const threats = this.dataStore.getThreats(this.filters);
      
      // Update map
      this.mapController.clearThreats();
      threats.forEach(threat => this.mapController.addThreat(threat));

      // Update analytics
      this.analyticsController.updateCharts(threats);

      // Update dashboard metrics
      $('#activeThreatCount').textContent = threats.length;
      $('#visibleCount').textContent = threats.length;
      $('#totalCount').textContent = this.dataStore.threats.length;
      $('#alertBadge').textContent = threats.filter(t => t.severity >= 4).length;

      // Update regional assessments
      const regions = {
        'North America': { count: 0, severity: 1 },
        'Eastern Europe': { count: 0, severity: 1 },
        'Middle East': { count: 0, severity: 1 },
        'Asia Pacific': { count: 0, severity: 1 }
      };

      threats.forEach(threat => {
        const region = threat.location.region;
        if (regions[region]) {
          regions[region].count++;
          regions[region].severity = Math.max(regions[region].severity, threat.severity);
        }
      });

      $$('.glass-panel.rounded-lg.p-4').forEach(panel => {
        const regionName = panel.querySelector('h4').textContent;
        const region = regions[regionName];
        if (region) {
          const countEl = panel.querySelector('.text-amber-400, .text-red-400, .text-green-400');
          const deltaEl = panel.querySelector('.text-red-400, .text-green-400');
          const statusEl = panel.querySelector('.text-xs.text-amber-400, .text-xs.text-red-400, .text-xs.text-green-400');
          const indicatorEl = panel.querySelector('.status-indicator');
          
          countEl.textContent = region.count;
          deltaEl.textContent = `+${Math.floor(Math.random() * 3)}`;
          statusEl.textContent = SEVERITY_LEVELS[region.severity].name;
          
          indicatorEl.className = 'status-indicator';
          indicatorEl.classList.add(region.severity >= 4 ? 'offline' : region.severity >= 3 ? 'maintenance' : 'online');
          
          panel.className = `glass-panel rounded-lg p-4 border border-${
            region.severity >= 5 ? 'red-700/50' :
            region.severity >= 4 ? 'red-500/30' :
            region.severity >= 3 ? 'amber-500/30' : 'green-500/30'
          }`;
        }
      });
    }

    startRealTimeUpdates() {
      setInterval(() => {
        if (this.paused) return;

        const categories = Object.keys(THREAT_CATEGORIES);
        const sources = ['satellite', 'osint', 'humint', 'sigint', 'ai'];
        const regions = [
          { lat: 40.7128, lng: -74.0060, country: 'United States', region: 'North America' },
          { lat: 48.3794, lng: 31.1656, country: 'Ukraine', region: 'Eastern Europe' },
          { lat: 24.4539, lng: 54.3773, country: 'UAE', region: 'Middle East' },
          { lat: 35.6762, lng: 139.6503, country: 'Japan', region: 'Asia Pacific' }
        ];

        const newThreat = {
          category: categories[Math.floor(Math.random() * categories.length)],
          severity: Math.floor(Math.random() * 5) + 1,
          title: `Threat ${Math.floor(Math.random() * 1000)} Detected`,
          description: `New ${THREAT_CATEGORIES[categories[Math.floor(Math.random() * categories.length)]].name} activity detected via ${sources[Math.floor(Math.random() * sources.length)]}.`,
          location: regions[Math.floor(Math.random() * regions.length)],
          source: sources[Math.floor(Math.random() * sources.length)],
          confidence: Math.floor(Math.random() * 30) + 70
        };

        this.dataStore.addThreat(newThreat);
      }, 10000);
    }
  }

  // Initialize application
  const app = new GlobalSecurityIntelligence();

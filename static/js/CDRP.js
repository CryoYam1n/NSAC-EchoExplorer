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

  // Advanced disaster categories for professional use
  const DISASTER_CATEGORIES = {
    flood: { color: '#3B82F6', icon: 'fas fa-water', name: 'Flood' },
    hurricane: { color: '#DC2626', icon: 'fas fa-wind', name: 'Hurricane' },
    wildfire: { color: '#D97706', icon: 'fas fa-fire', name: 'Wildfire' },
    earthquake: { color: '#8B5CF6', icon: 'fas fa-house-damage', name: 'Earthquake' },
    drought: { color: '#B91C1C', icon: 'fas fa-sun', name: 'Drought' }
  };

  const SEVERITY_LEVELS = {
    1: { name: 'INFO', color: '#6B7280', priority: 'Informational' },
    2: { name: 'LOW', color: '#059669', priority: 'Low Priority' },
    3: { name: 'MEDIUM', color: '#D97706', priority: 'Medium Priority' },
    4: { name: 'HIGH', color: '#DC2626', priority: 'High Priority' },
    5: { name: 'CRITICAL', color: '#B91C1C', priority: 'Critical Priority' }
  };

  // Data Store
  class DisasterDataStore {
    constructor() {
      this.disasters = [];
      this.subscribers = [];
    }

    addDisaster(disaster) {
      disaster.id = crypto.randomUUID();
      disaster.timestamp = Date.now();
      this.disasters.unshift(disaster);
      this.notify('add', disaster);
    }

    updateDisaster(id, updates) {
      const index = this.disasters.findIndex(t => t.id === id);
      if (index !== -1) {
        this.disasters[index] = { ...this.disasters[index], ...updates };
        this.notify('update', this.disasters[index]);
      }
    }

    removeDisaster(id) {
      const index = this.disasters.findIndex(t => t.id === id);
      if (index !== -1) {
        const disaster = this.disasters.splice(index, 1)[0];
        this.notify('remove', disaster);
      }
    }

    getDisasters(filters = {}) {
      return this.disasters.filter(disaster => {
        if (filters.category && filters.category !== 'all' && disaster.category !== filters.category) return false;
        if (filters.severity && disaster.severity < filters.severity) return false;
        if (filters.timeWindow) {
          const cutoff = Date.now() - (filters.timeWindow * 60 * 60 * 1000);
          if (disaster.timestamp < cutoff) return false;
        }
        if (filters.source && filters.source !== 'all' && disaster.source !== filters.source) return false;
        if (filters.search) {
          const searchLower = filters.search.toLowerCase();
          const searchableText = `${disaster.title} ${disaster.description} ${disaster.location?.country || ''} ${disaster.location?.region || ''}`.toLowerCase();
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

    notify(action, disaster) {
      this.subscribers.forEach(callback => callback(action, disaster));
    }

    clear() {
      this.disasters = [];
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

      this.disasterMarkers = new Map();
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
      this.disasterMarkers = new Map();

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
        if (marker.options.disaster && marker.options.disaster.severity > maxSeverity) {
          maxSeverity = marker.options.disaster.severity;
        }
      });
      return maxSeverity;
    }

    createDisasterMarker(disaster) {
      const category = DISASTER_CATEGORIES[disaster.category] || DISASTER_CATEGORIES.flood;
      const severity = SEVERITY_LEVELS[disaster.severity] || SEVERITY_LEVELS[1];
      
      const icon = L.divIcon({
        html: `
          <div class="disaster-marker severity-${disaster.severity}" style="background-color: ${category.color}">
            <i class="${category.icon}"></i>
            ${disaster.severity >= 4 ? '<div class="pulse-ring"></div>' : ''}
          </div>
        `,
        className: 'custom-disaster-marker',
        iconSize: [24, 24],
        iconAnchor: [12, 12]
      });

      const marker = L.marker([disaster.location.lat, disaster.location.lng], {
        icon,
        disaster
      });

      marker.bindPopup(`
        <div class="disaster-popup">
          <div class="popup-header">
            <div class="disaster-category">
              <i class="${category.icon}"></i>
              ${category.name}
            </div>
            <div class="severity-badge severity-${disaster.severity}">
              ${severity.name}
            </div>
          </div>
          <h3>${disaster.title}</h3>
          <p>${disaster.description}</p>
          <div class="popup-meta">
            <div class="meta-item">
              <i class="fas fa-map-marker-alt"></i>
              ${disaster.location.country}, ${disaster.location.region}
            </div>
            <div class="meta-item">
              <i class="fas fa-clock"></i>
              ${formatTime(disaster.timestamp)} UTC
            </div>
            <div class="meta-item">
              <i class="fas fa-database"></i>
              Source: ${disaster.source}
            </div>
            <div class="meta-item">
              <i class="fas fa-percent"></i>
              Confidence: ${disaster.confidence}%
            </div>
          </div>
          <div class="popup-actions">
            <button class="btn-details" data-disaster-id="${disaster.id}">
              <i class="fas fa-info-circle"></i> Details
            </button>
            <button class="btn-track" data-disaster-id="${disaster.id}">
              <i class="fas fa-crosshairs"></i> Track
            </button>
          </div>
        </div>
      `, { maxWidth: 350 });

      return marker;
    }

    addDisaster(disaster) {
      if (this.disasterMarkers.has(disaster.id)) return;
      
      const marker = this.createDisasterMarker(disaster);
      this.disasterMarkers.set(disaster.id, marker);
      
      if (this.useCluster) {
        this.clusterGroup.addLayer(marker);
      } else {
        marker.addTo(this.map);
      }
      
      this.updateHeatmap();
    }

    updateDisaster(disaster) {
      if (this.disasterMarkers.has(disaster.id)) {
        this.removeDisaster(disaster.id);
        this.addDisaster(disaster);
      }
    }

    removeDisaster(disasterId) {
      const marker = this.disasterMarkers.get(disasterId);
      if (marker) {
        if (this.useCluster) {
          this.clusterGroup.removeLayer(marker);
        } else {
          this.map.removeLayer(marker);
        }
        this.disasterMarkers.delete(disasterId);
        this.updateHeatmap();
      }
    }

    clearDisasters() {
      this.disasterMarkers.forEach(marker => {
        if (this.useCluster) {
          this.clusterGroup.removeLayer(marker);
        } else {
          this.map.removeLayer(marker);
        }
      });
      this.disasterMarkers.clear();
      this.updateHeatmap();
    }

    toggleCluster(enabled) {
      this.useCluster = enabled;
      
      if (enabled) {
        this.disasterMarkers.forEach(marker => {
          this.map.removeLayer(marker);
          this.clusterGroup.addLayer(marker);
        });
        this.map.addLayer(this.clusterGroup);
      } else {
        this.map.removeLayer(this.clusterGroup);
        this.disasterMarkers.forEach(marker => {
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
      
      const heatPoints = Array.from(this.disasterMarkers.values()).map(marker => {
        const disaster = marker.options.disaster;
        return [
          disaster.location.lat,
          disaster.location.lng,
          disaster.severity / 5 // Normalize severity to 0-1 range
        ];
      });
      
      this.heatmapLayer.setLatLngs(heatPoints);
    }

    initializeMapControls() {
      // Custom CSS for map elements
      const style = document.createElement('style');
      style.textContent = `
        .custom-disaster-marker {
          background: none;
          border: none;
        }
        
        .disaster-marker {
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
        
        .disaster-marker i {
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
        
        .disaster-popup {
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
        
        .disaster-category {
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
        
        .disaster-popup h3 {
          margin: 0 0 8px 0;
          font-size: 16px;
          font-weight: 600;
          color: white;
        }
        
        .disaster-popup p {
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

    focusOnDisaster(disasterId) {
      const marker = this.disasterMarkers.get(disasterId);
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
            label: 'Disaster Activity',
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

      // Disaster Matrix Chart
      this.disasterMatrixChart = new Chart($('#disasterMatrix'), {
        type: 'doughnut',
        data: {
          labels: Object.keys(DISASTER_CATEGORIES).map(key => DISASTER_CATEGORIES[key].name),
          datasets: [{
            data: [],
            backgroundColor: Object.keys(DISASTER_CATEGORIES).map(key => DISASTER_CATEGORIES[key].color),
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

    updateCharts(disasters) {
      this.updateTimelineChart(disasters);
      this.updateDisasterMatrixChart(disasters);
    }

    updateTimelineChart(disasters) {
      const now = Date.now();
      const hours = 24;
      const timeSlots = [];
      
      for (let i = hours - 1; i >= 0; i--) {
        timeSlots.push(new Date(now - (i * 60 * 60 * 1000)));
      }
      
      const counts = timeSlots.map(slot => {
        const slotStart = slot.getTime();
        const slotEnd = slotStart + (60 * 60 * 1000);
        return disasters.filter(t => t.timestamp >= slotStart && t.timestamp < slotEnd).length;
      });

      this.timelineChart.data.labels = timeSlots;
      this.timelineChart.data.datasets[0].data = counts;
      this.timelineChart.update('none');
    }

    updateDisasterMatrixChart(disasters) {
      const categoryCounts = {};
      Object.keys(DISASTER_CATEGORIES).forEach(key => {
        categoryCounts[key] = disasters.filter(t => t.category === key).length;
      });

      this.disasterMatrixChart.data.datasets[0].data = Object.values(categoryCounts);
      this.disasterMatrixChart.update('none');
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
  class ClimateDisasterRisk {
    constructor() {
      this.dataStore = new DisasterDataStore();
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
      $('#disasterFilter').addEventListener('change', (e) => {
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
            $('#zoomInBtn').addEventListener('click', () => {
        this.mapController.map.zoomIn();
      });

      $('#zoomOutBtn').addEventListener('click', () => {
        this.mapController.map.zoomOut();
      });

      $('#locateBtn').addEventListener('click', () => {
        if (navigator.geolocation) {
          navigator.geolocation.getCurrentPosition(
            (position) => {
              const { latitude, longitude } = position.coords;
              this.mapController.map.setView([latitude, longitude], 12);
              L.marker([latitude, longitude], {
                icon: L.divIcon({
                  html: '<div class="disaster-marker" style="background-color: #3B82F6"><i class="fas fa-map-pin"></i></div>',
                  className: 'custom-disaster-marker',
                  iconSize: [24, 24],
                  iconAnchor: [12, 12]
                })
              }).addTo(this.mapController.map)
                .bindPopup('Your Location')
                .openPopup();
            },
            () => {
              this.notificationSystem.show({
                title: 'Location Error',
                body: 'Unable to access location. Please check permissions.'
              }, 'error');
            }
          );
        } else {
          this.notificationSystem.show({
            title: 'Location Error',
            body: 'Geolocation not supported by your browser.'
          }, 'error');
        }
      });

      $('#satelliteLayerBtn').addEventListener('click', (e) => {
        if (this.mapController.labelsLayer.isAdded) {
          this.mapController.map.removeLayer(this.mapController.labelsLayer);
          e.target.querySelector('i').classList.replace('text-blue-400', 'text-slate-400');
          this.mapController.labelsLayer.isAdded = false;
        } else {
          this.mapController.map.addLayer(this.mapController.labelsLayer);
          e.target.querySelector('i').classList.replace('text-slate-400', 'text-blue-400');
          this.mapController.labelsLayer.isAdded = true;
        }
      });

      $('#weatherLayerBtn').addEventListener('click', (e) => {
        // Placeholder for weather layer toggle (API integration would go here)
        this.notificationSystem.show({
          title: 'Weather Layer',
          body: 'Weather overlay not implemented in this version.'
        }, 'info');
      });

      // Sidebar controls
      $('#sidebarToggle').addEventListener('click', () => {
        $('.sidebar').classList.toggle('open');
      });

      $('#fullscreenBtn').addEventListener('click', () => {
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen();
        } else {
          document.exitFullscreen();
        }
      });

      // Alert controls
      $('#alertsBtn').addEventListener('click', () => {
        $('#alertModal').classList.add('active');
        this.renderAlertModal();
      });

      $('#closeAlertModal').addEventListener('click', () => {
        $('#alertModal').classList.remove('active');
      });

      // Feed controls
      $('#pauseFeedBtn').addEventListener('click', (e) => {
        const isPaused = e.target.querySelector('i').classList.contains('fa-play');
        if (isPaused) {
          e.target.innerHTML = '<i class="fas fa-pause"></i> PAUSE';
          this.startRealTimeUpdates();
        } else {
          e.target.innerHTML = '<i class="fas fa-play"></i> RESUME';
          this.stopRealTimeUpdates();
        }
      });

      $('#clearFeedBtn').addEventListener('click', () => {
        this.dataStore.clear();
        $('#intelligenceFeed').innerHTML = '';
        this.notificationSystem.show({
          title: 'Feed Cleared',
          body: 'Intelligence feed has been reset.'
        }, 'success');
      });

      // Disaster matrix filter
      $$('.disaster-cell').forEach(cell => {
        cell.addEventListener('click', (e) => {
          const level = e.target.dataset.level;
          $('#severityFilter').value = {
            low: 2,
            medium: 3,
            high: 4,
            critical: 5
          }[level] || 1;
          this.filters.severity = parseInt($('#severityFilter').value);
          this.updateDashboard();
        });
      });

      // Map popup actions
      this.mapController.map.on('popupopen', (e) => {
        const popup = e.popup._contentNode;
        popup.querySelectorAll('.btn-details').forEach(btn => {
          btn.addEventListener('click', () => {
            const disasterId = btn.dataset.disasterId;
            $('#alertModal').classList.add('active');
            this.renderAlertModal(disasterId);
          });
        });

        popup.querySelectorAll('.btn-track').forEach(btn => {
          btn.addEventListener('click', () => {
            const disasterId = btn.dataset.disasterId;
            this.mapController.focusOnDisaster(disasterId);
            this.notificationSystem.show({
              title: 'Tracking Enabled',
              body: `Now tracking disaster ID: ${disasterId.substring(0, 8)}`
            }, 'success');
          });
        });
      });

      // Data store subscription
      this.dataStore.subscribe((action, disaster) => {
        if (action === 'add') {
          this.mapController.addDisaster(disaster);
          this.addToFeed(disaster);
          if (disaster.severity >= 4) {
            this.notificationSystem.show({
              title: `New ${SEVERITY_LEVELS[disaster.severity].name} Disaster`,
              body: disaster.title
            }, disaster.severity === 5 ? 'critical' : 'error');
          }
        } else if (action === 'update') {
          this.mapController.updateDisaster(disaster);
          this.updateFeedItem(disaster);
        } else if (action === 'remove') {
          this.mapController.removeDisaster(disaster.id);
          this.removeFeedItem(disaster.id);
        } else if (action === 'clear') {
          this.mapController.clearDisasters();
          $('#intelligenceFeed').innerHTML = '';
        }
        this.updateDashboard();
      });
    }

    startTimeUpdate() {
      const updateTime = () => {
        const now = new Date();
        $('#utcTime').textContent = formatTime(now);
        $('#lastUpdate').textContent = formatTime(now);
      };
      updateTime();
      setInterval(updateTime, 1000);
    }

    startRealTimeUpdates() {
      if (this.updateInterval) return;
      this.updateInterval = setInterval(() => {
        this.generateSampleData();
        this.updateDashboard();
      }, 15000);
    }

    stopRealTimeUpdates() {
      if (this.updateInterval) {
        clearInterval(this.updateInterval);
        this.updateInterval = null;
      }
    }

    generateSampleData() {
      const sampleDisasters = [
  {
    category: 'wildfire',
    severity: 4,
    title: 'California Wildfire Outbreak',
    description: 'Multiple wildfires reported in Northern California, threatening residential areas and forests.',
    location: { lat: 39.7, lng: -122.4, country: 'United States', region: 'California' },
    source: 'satellite',
    confidence: 95
  },
  {
    category: 'hurricane',
    severity: 5,
    title: 'Hurricane Delta Approaching Gulf Coast',
    description: 'Category 4 hurricane expected to make landfall in Louisiana with severe storm surges.',
    location: { lat: 29.5, lng: -90.3, country: 'United States', region: 'Gulf Coast' },
    source: 'climate',
    confidence: 98
  },
  {
    category: 'flood',
    severity: 3,
    title: 'Monsoon Flooding in Mumbai',
    description: 'Heavy rainfall causing widespread flooding in urban areas, disrupting transportation.',
    location: { lat: 19.1, lng: 72.8, country: 'India', region: 'Maharashtra' },
    source: 'ground',
    confidence: 90
  },
  {
    category: 'drought',
    severity: 4,
    title: 'Severe Drought in Amazon Basin',
    description: 'Prolonged drought impacting agriculture and water supply in the Amazon rainforest.',
    location: { lat: -3.4, lng: -62.2, country: 'Brazil', region: 'Amazonas' },
    source: 'satellite',
    confidence: 92
  },
  {
    category: 'earthquake',
    severity: 2,
    title: 'Minor Earthquake in Hokkaido',
    description: '4.8 magnitude earthquake reported off the coast, no significant damage.',
    location: { lat: 43.1, lng: 141.4, country: 'Japan', region: 'Hokkaido' },
    source: 'ground',
    confidence: 85
  },
  {
    category: 'wildfire',
    severity: 3,
    title: 'Australian Bushfires in New South Wales',
    description: 'Bushfires spreading across rural areas, threatening wildlife and farmland.',
    location: { lat: -33.8, lng: 151.0, country: 'Australia', region: 'New South Wales' },
    source: 'satellite',
    confidence: 93
  },
  {
    category: 'hurricane',
    severity: 4,
    title: 'Hurricane Maria in Caribbean',
    description: 'Category 3 hurricane approaching Puerto Rico, high risk of flooding.',
    location: { lat: 18.2, lng: -66.5, country: 'Puerto Rico', region: 'Caribbean' },
    source: 'climate',
    confidence: 96
  },
  {
    category: 'flood',
    severity: 4,
    title: 'Yangtze River Flooding',
    description: 'Severe flooding along the Yangtze River, displacing thousands.',
    location: { lat: 30.6, lng: 114.3, country: 'China', region: 'Hubei' },
    source: 'ground',
    confidence: 91
  },
  {
    category: 'drought',
    severity: 3,
    title: 'Sahel Region Drought',
    description: 'Ongoing drought affecting crops and livestock in the Sahel belt.',
    location: { lat: 14.5, lng: 0.0, country: 'Mali', region: 'Sahel' },
    source: 'satellite',
    confidence: 89
  },
  {
    category: 'earthquake',
    severity: 3,
    title: 'Moderate Earthquake in Peru',
    description: '5.6 magnitude earthquake in Andes, minor structural damage reported.',
    location: { lat: -13.5, lng: -72.5, country: 'Peru', region: 'Cusco' },
    source: 'ground',
    confidence: 87
  },
  {
    category: 'wildfire',
    severity: 5,
    title: 'Siberian Forest Fires',
    description: 'Massive wildfires burning through Siberian taiga, releasing significant CO2.',
    location: { lat: 60.0, lng: 105.0, country: 'Russia', region: 'Siberia' },
    source: 'satellite',
    confidence: 97
  },
  {
    category: 'hurricane',
    severity: 3,
    title: 'Tropical Storm in Philippines',
    description: 'Tropical storm causing heavy rains and landslides in Luzon.',
    location: { lat: 14.6, lng: 121.0, country: 'Philippines', region: 'Luzon' },
    source: 'climate',
    confidence: 94
  },
  {
    category: 'flood',
    severity: 2,
    title: 'Coastal Flooding in Bangladesh',
    description: 'Minor coastal flooding due to high tides and monsoon rains.',
    location: { lat: 22.7, lng: 90.4, country: 'Bangladesh', region: 'Chittagong' },
    source: 'ground',
    confidence: 88
  },
  {
    category: 'drought',
    severity: 5,
    title: 'Horn of Africa Drought',
    description: 'Extreme drought leading to famine risks in Somalia and Ethiopia.',
    location: { lat: 9.5, lng: 47.0, country: 'Somalia', region: 'Horn of Africa' },
    source: 'satellite',
    confidence: 96
  },
  {
    category: 'earthquake',
    severity: 4,
    title: 'Earthquake in Indonesia',
    description: '6.2 magnitude earthquake near Sumatra, tsunami warning issued.',
    location: { lat: 3.3, lng: 95.8, country: 'Indonesia', region: 'Aceh' },
    source: 'ground',
    confidence: 90
  },
  {
    category: 'wildfire',
    severity: 3,
    title: 'Wildfires in Greece',
    description: 'Fires burning near Athens, prompting evacuations.',
    location: { lat: 37.9, lng: 23.7, country: 'Greece', region: 'Attica' },
    source: 'satellite',
    confidence: 92
  },
  {
    category: 'hurricane',
    severity: 5,
    title: 'Hurricane Fiona in Atlantic',
    description: 'Category 4 hurricane approaching Bermuda with high winds.',
    location: { lat: 32.3, lng: -64.7, country: 'Bermuda', region: 'Atlantic' },
    source: 'climate',
    confidence: 97
  },
  {
    category: 'flood',
    severity: 4,
    title: 'Nile River Flooding',
    description: 'Seasonal flooding impacting agriculture in Sudan.',
    location: { lat: 15.6, lng: 32.5, country: 'Sudan', region: 'Khartoum' },
    source: 'ground',
    confidence: 90
  },
  {
    category: 'drought',
    severity: 3,
    title: 'California Central Valley Drought',
    description: 'Ongoing drought affecting water reservoirs and agriculture.',
    location: { lat: 36.7, lng: -119.7, country: 'United States', region: 'California' },
    source: 'satellite',
    confidence: 91
  },
  {
    category: 'earthquake',
    severity: 2,
    title: 'Minor Earthquake in Chile',
    description: '4.5 magnitude earthquake near Santiago, no damage reported.',
    location: { lat: -33.5, lng: -70.6, country: 'Chile', region: 'Santiago' },
    source: 'ground',
    confidence: 86
  },
  {
    category: 'wildfire',
    severity: 4,
    title: 'Amazon Wildfires',
    description: 'Fires in Mato Grosso threatening biodiversity.',
    location: { lat: -15.5, lng: -55.0, country: 'Brazil', region: 'Mato Grosso' },
    source: 'satellite',
    confidence: 94
  },
  {
    category: 'hurricane',
    severity: 4,
    title: 'Hurricane Ian in Cuba',
    description: 'Category 3 hurricane causing power outages and flooding.',
    location: { lat: 22.0, lng: -78.0, country: 'Cuba', region: 'Pinar del Rio' },
    source: 'climate',
    confidence: 95
  },
  {
    category: 'flood',
    severity: 3,
    title: 'Flooding in Germany',
    description: 'Heavy rains causing river overflows in Bavaria.',
    location: { lat: 48.1, lng: 11.6, country: 'Germany', region: 'Bavaria' },
    source: 'ground',
    confidence: 89
  },
  {
    category: 'drought',
    severity: 4,
    title: 'Southern Europe Drought',
    description: 'Prolonged heatwave and drought affecting Spain and Portugal.',
    location: { lat: 40.4, lng: -3.7, country: 'Spain', region: 'Madrid' },
    source: 'satellite',
    confidence: 93
  },
  {
    category: 'earthquake',
    severity: 3,
    title: 'Earthquake in Turkey',
    description: '5.4 magnitude earthquake near Izmir, minor injuries reported.',
    location: { lat: 38.4, lng: 27.1, country: 'Turkey', region: 'Izmir' },
    source: 'ground',
    confidence: 88
  },
  {
    category: 'wildfire',
    severity: 3,
    title: 'Wildfires in Alberta',
    description: 'Forest fires spreading in Canada’s boreal forests.',
    location: { lat: 53.9, lng: -116.0, country: 'Canada', region: 'Alberta' },
    source: 'satellite',
    confidence: 92
  },
  {
    category: 'hurricane',
    severity: 3,
    title: 'Tropical Storm in Vietnam',
    description: 'Storm causing coastal flooding in central Vietnam.',
    location: { lat: 16.0, lng: 108.2, country: 'Vietnam', region: 'Da Nang' },
    source: 'climate',
    confidence: 94
  },
  {
    category: 'flood',
    severity: 4,
    title: 'Mississippi River Flooding',
    description: 'Major flooding along the Mississippi, impacting farmland.',
    location: { lat: 35.1, lng: -90.0, country: 'United States', region: 'Mississippi' },
    source: 'ground',
    confidence: 91
  },
  {
    category: 'drought',
    severity: 3,
    title: 'South African Drought',
    description: 'Drought affecting water supply in Cape Town region.',
    location: { lat: -33.9, lng: 18.4, country: 'South Africa', region: 'Western Cape' },
    source: 'satellite',
    confidence: 90
  },
  {
    category: 'earthquake',
    severity: 4,
    title: 'Earthquake in Mexico',
    description: '6.0 magnitude earthquake near Oaxaca, structural damage reported.',
    location: { lat: 16.9, lng: -95.0, country: 'Mexico', region: 'Oaxaca' },
    source: 'ground',
    confidence: 89
  },
  {
    category: 'wildfire',
    severity: 5,
    title: 'Australian Outback Fires',
    description: 'Massive wildfires in Northern Territory, threatening remote communities.',
    location: { lat: -23.7, lng: 133.9, country: 'Australia', region: 'Northern Territory' },
    source: 'satellite',
    confidence: 96
  },
  {
    category: 'hurricane',
    severity: 4,
    title: 'Hurricane in Bahamas',
    description: 'Category 3 hurricane causing widespread damage in Nassau.',
    location: { lat: 25.0, lng: -77.4, country: 'Bahamas', region: 'Nassau' },
    source: 'climate',
    confidence: 95
  },
  {
    category: 'flood',
    severity: 3,
    title: 'Flooding in Nigeria',
    description: 'Seasonal flooding in Lagos, displacing residents.',
    location: { lat: 6.5, lng: 3.4, country: 'Nigeria', region: 'Lagos' },
    source: 'ground',
    confidence: 90
  },
  {
    category: 'drought',
    severity: 4,
    title: 'Central Asia Drought',
    description: 'Severe drought affecting Uzbekistan’s agriculture.',
    location: { lat: 41.3, lng: 69.2, country: 'Uzbekistan', region: 'Tashkent' },
    source: 'satellite',
    confidence: 92
  },
  {
    category: 'earthquake',
    severity: 2,
    title: 'Minor Earthquake in New Zealand',
    description: '4.7 magnitude earthquake near Wellington, no damage.',
    location: { lat: -41.3, lng: 174.8, country: 'New Zealand', region: 'Wellington' },
    source: 'ground',
    confidence: 86
  },
  {
    category: 'wildfire',
    severity: 3,
    title: 'Fires in Portugal',
    description: 'Wildfires in Algarve region, threatening tourism areas.',
    location: { lat: 37.1, lng: -8.7, country: 'Portugal', region: 'Algarve' },
    source: 'satellite',
    confidence: 91
  },
  {
    category: 'hurricane',
    severity: 5,
    title: 'Hurricane in Florida',
    description: 'Category 4 hurricane approaching Miami with severe flooding risks.',
    location: { lat: 25.8, lng: -80.2, country: 'United States', region: 'Florida' },
    source: 'climate',
    confidence: 97
  },
  {
    category: 'flood',
    severity: 3,
    title: 'Flooding in Thailand',
    description: 'Monsoon flooding in Bangkok, impacting infrastructure.',
    location: { lat: 13.7, lng: 100.5, country: 'Thailand', region: 'Bangkok' },
    source: 'ground',
    confidence: 89
  },
  {
    category: 'drought',
    severity: 3,
    title: 'Drought in Argentina',
    description: 'Drought affecting soybean production in Pampas region.',
    location: { lat: -34.6, lng: -58.4, country: 'Argentina', region: 'Buenos Aires' },
    source: 'satellite',
    confidence: 90
  },
  {
    category: 'earthquake',
    severity: 4,
    title: 'Earthquake in Philippines',
    description: '6.1 magnitude earthquake near Davao, minor damage reported.',
    location: { lat: 7.1, lng: 125.6, country: 'Philippines', region: 'Davao' },
    source: 'ground',
    confidence: 88
  },
  {
    category: 'wildfire',
    severity: 4,
    title: 'Fires in Bolivia',
    description: 'Wildfires in Santa Cruz region, impacting forests and farmland.',
    location: { lat: -16.5, lng: -60.7, country: 'Bolivia', region: 'Santa Cruz' },
    source: 'satellite',
    confidence: 93
  },
  {
    category: 'hurricane',
    severity: 3,
    title: 'Tropical Storm in Mexico',
    description: 'Storm causing heavy rains in Yucatan Peninsula.',
    location: { lat: 20.6, lng: -88.2, country: 'Mexico', region: 'Yucatan' },
    source: 'climate',
    confidence: 94
  },
  {
    category: 'flood',
    severity: 4,
    title: 'Flooding in Pakistan',
    description: 'Monsoon flooding in Sindh, displacing thousands.',
    location: { lat: 24.9, lng: 67.1, country: 'Pakistan', region: 'Sindh' },
    source: 'ground',
    confidence: 91
  },
  {
    category: 'drought',
    severity: 4,
    title: 'Drought in Ethiopia',
    description: 'Severe drought in Oromia, threatening food security.',
    location: { lat: 9.1, lng: 38.7, country: 'Ethiopia', region: 'Oromia' },
    source: 'satellite',
    confidence: 92
  },
  {
    category: 'earthquake',
    severity: 3,
    title: 'Earthquake in Italy',
    description: '5.3 magnitude earthquake in Sicily, minor disruptions.',
    location: { lat: 37.6, lng: 15.1, country: 'Italy', region: 'Sicily' },
    source: 'ground',
    confidence: 87
  },
  {
    category: 'wildfire',
    severity: 3,
    title: 'Fires in Chile',
    description: 'Wildfires in Valparaiso region, threatening coastal areas.',
    location: { lat: -33.0, lng: -71.6, country: 'Chile', region: 'Valparaiso' },
    source: 'satellite',
    confidence: 92
  },
  {
    category: 'hurricane',
    severity: 4,
    title: 'Hurricane in Jamaica',
    description: 'Category 3 hurricane causing flooding and power outages.',
    location: { lat: 18.0, lng: -76.8, country: 'Jamaica', region: 'Kingston' },
    source: 'climate',
    confidence: 95
  },
  {
    category: 'flood',
    severity: 3,
    title: 'Flooding in Vietnam',
    description: 'River flooding in Mekong Delta, affecting rice production.',
    location: { lat: 10.0, lng: 105.0, country: 'Vietnam', region: 'Mekong Delta' },
    source: 'ground',
    confidence: 90
  },
  {
    category: 'drought',
    severity: 3,
    title: 'Drought in Morocco',
    description: 'Drought impacting agriculture in Atlas Mountains.',
    location: { lat: 31.6, lng: -7.0, country: 'Morocco', region: 'Marrakech' },
    source: 'satellite',
    confidence: 91
  },
  {
    category: 'earthquake',
    severity: 2,
    title: 'Minor Earthquake in Alaska',
    description: '4.6 magnitude earthquake near Anchorage, no significant impact.',
    location: { lat: 61.2, lng: -149.9, country: 'United States', region: 'Alaska' },
    source: 'ground',
    confidence: 86
  }
];

      const randomDisaster = sampleDisasters[Math.floor(Math.random() * sampleDisasters.length)];
      this.dataStore.addDisaster({ ...randomDisaster, timestamp: Date.now() });
    }
    generateSampleData() {
  const disasterTemplates = [
    {
      category: 'wildfire',
      titles: [
        '{region} Wildfire Outbreak',
        'Forest Fires in {region}',
        '{region} Bushfires Threatening Communities'
      ],
      descriptions: [
        'Wildfires spreading rapidly in {region}, threatening {impact}.',
        'Multiple fires reported in {region}, impacting {impact}.',
        'Uncontrolled wildfires in {region}, causing {impact}.'
      ],
      impacts: ['residential areas', 'wildlife habitats', 'forests and farmland'],
      source: 'satellite',
      severityRange: [3, 5],
      confidenceRange: [90, 97]
    },
    {
      category: 'hurricane',
      titles: [
        'Hurricane {name} in {region}',
        'Tropical Storm Approaching {region}',
        'Category {severity} Hurricane in {region}'
      ],
      descriptions: [
        'Hurricane {name} expected to cause {impact} in {region}.',
        'Tropical storm bringing heavy rains and {impact} to {region}.',
        'Major hurricane impacting {region} with {impact}.'
      ],
      impacts: ['severe flooding', 'power outages', 'storm surges'],
      source: 'climate',
      severityRange: [3, 5],
      confidenceRange: [94, 98]
    },
    {
      category: 'flood',
      titles: [
        '{region} Flooding Event',
        'Monsoon Floods in {region}',
        'River Overflow in {region}'
      ],
      descriptions: [
        'Heavy rainfall causing flooding in {region}, affecting {impact}.',
        'Floods in {region} displacing residents and damaging {impact}.',
        'River flooding impacting {region}, disrupting {impact}.'
      ],
      impacts: ['urban infrastructure', 'agriculture', 'transportation'],
      source: 'ground',
      severityRange: [2, 4],
      confidenceRange: [88, 92]
    },
    {
      category: 'drought',
      titles: [
        'Severe Drought in {region}',
        '{region} Water Crisis',
        'Prolonged Drought in {region}'
      ],
      descriptions: [
        'Ongoing drought in {region} affecting {impact}.',
        'Severe water shortage in {region}, impacting {impact}.',
        'Drought conditions in {region} threatening {impact}.'
      ],
      impacts: ['agriculture', 'water supply', 'food security'],
      source: 'satellite',
      severityRange: [3, 5],
      confidenceRange: [90, 94]
    },
    {
      category: 'earthquake',
      titles: [
        '{severityText} Earthquake in {region}',
        'Seismic Activity Near {region}',
        'Earthquake Reported in {region}'
      ],
      descriptions: [
        '{magnitude} magnitude earthquake in {region}, causing {impact}.',
        'Earthquake near {region} with {impact}.',
        'Seismic event in {region}, resulting in {impact}.'
      ],
      impacts: ['no significant damage', 'minor structural damage', 'temporary disruptions'],
      source: 'ground',
      severityRange: [2, 4],
      confidenceRange: [85, 90]
    }
  ];

  const locations = [
    { country: 'United States', region: 'California', lat: 39.7, lng: -122.4 },
    { country: 'United States', region: 'Gulf Coast', lat: 29.5, lng: -90.3 },
    { country: 'India', region: 'Maharashtra', lat: 19.1, lng: 72.8 },
    { country: 'Brazil', region: 'Amazonas', lat: -3.4, lng: -62.2 },
    { country: 'Japan', region: 'Hokkaido', lat: 43.1, lng: 141.4 },
    { country: 'Australia', region: 'New South Wales', lat: -33.8, lng: 151.0 },
    { country: 'Puerto Rico', region: 'Caribbean', lat: 18.2, lng: -66.5 },
    { country: 'China', region: 'Hubei', lat: 30.6, lng: 114.3 },
    { country: 'Mali', region: 'Sahel', lat: 14.5, lng: 0.0 },
    { country: 'Peru', region: 'Cusco', lat: -13.5, lng: -72.5 },
    { country: 'Russia', region: 'Siberia', lat: 60.0, lng: 105.0 },
    { country: 'Philippines', region: 'Luzon', lat: 14.6, lng: 121.0 },
    { country: 'Bangladesh', region: 'Chittagong', lat: 22.7, lng: 90.4 },
    { country: 'Somalia', region: 'Horn of Africa', lat: 9.5, lng: 47.0 },
    { country: 'Indonesia', region: 'Aceh', lat: 3.3, lng: 95.8 }
    // Add more locations as needed
  ];

  const hurricaneNames = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa'];
  const getRandomElement = (arr) => arr[Math.floor(Math.random() * arr.length)];
  const getRandomInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;

  const template = getRandomElement(disasterTemplates);
  const location = getRandomElement(locations);
  const severity = getRandomInt(template.severityRange[0], template.severityRange[1]);
  const confidence = getRandomInt(template.confidenceRange[0], template.confidenceRange[1]);
  const impact = getRandomElement(template.impacts);
  let title = getRandomElement(template.titles);
  let description = getRandomElement(template.descriptions);

  // Customize title and description
  title = title.replace('{region}', location.region);
  if (template.category === 'hurricane') {
    title = title.replace('{name}', getRandomElement(hurricaneNames));
    title = title.replace('{severity}', severity - 1);
  }
  if (template.category === 'earthquake') {
    const magnitude = (4.0 + (severity - 2) * 1.5).toFixed(1);
    description = description.replace('{magnitude}', magnitude);
  }
  description = description.replace('{region}', location.region).replace('{impact}', impact);
  if (template.category === 'earthquake') {
    description = description.replace('{severityText}', severity >= 4 ? 'Major' : severity === 3 ? 'Moderate' : 'Minor');
  }

  const disaster = {
    category: template.category,
    severity,
    title,
    description,
    location: { lat: location.lat, lng: location.lng, country: location.country, region: location.region },
    source: template.source,
    confidence,
    timestamp: Date.now()
  };

  this.dataStore.addDisaster(disaster);
}

    addToFeed(disaster) {
      const category = DISASTER_CATEGORIES[disaster.category] || DISASTER_CATEGORIES.flood;
      const severity = SEVERITY_LEVELS[disaster.severity] || SEVERITY_LEVELS[1];
      
      const feedItem = document.createElement('div');
      feedItem.className = 'glass-panel rounded-lg p-4 border-l-4';
      feedItem.style.borderColor = category.color;
      feedItem.dataset.disasterId = disaster.id;
      feedItem.innerHTML = `
        <div class="flex items-start gap-3">
          <i class="${category.icon} text-lg" style="color: ${category.color}"></i>
          <div class="flex-1">
            <div class="flex items-center justify-between">
              <div class="text-sm font-medium text-white">${disaster.title}</div>
              <div class="text-xs text-slate-400 terminal-font">${timeAgo(disaster.timestamp)}</div>
            </div>
            <div class="text-xs text-slate-300 mt-1">${disaster.description}</div>
            <div class="flex items-center justify-between mt-2 text-xs">
              <div class="flex items-center gap-2">
                <span class="severity-badge severity-${disaster.severity}">${severity.name}</span>
                <span class="text-slate-400">${disaster.location.country}</span>
              </div>
              <div class="flex items-center gap-2">
                <button class="btn-details text-blue-400 hover:text-blue-300" data-disaster-id="${disaster.id}">
                  <i class="fas fa-info-circle"></i> Details
                </button>
                <button class="btn-track text-green-400 hover:text-green-300" data-disaster-id="${disaster.id}">
                  <i class="fas fa-crosshairs"></i> Track
                </button>
              </div>
            </div>
          </div>
        </div>
      `;

      $('#intelligenceFeed').prepend(feedItem);

      feedItem.querySelector('.btn-details').addEventListener('click', () => {
        $('#alertModal').classList.add('active');
        this.renderAlertModal(disaster.id);
      });

      feedItem.querySelector('.btn-track').addEventListener('click', () => {
        this.mapController.focusOnDisaster(disaster.id);
        this.notificationSystem.show({
          title: 'Tracking Enabled',
          body: `Now tracking disaster ID: ${disaster.id.substring(0, 8)}`
        }, 'success');
      });
    }

    updateFeedItem(disaster) {
      const feedItem = $(`[data-disaster-id="${disaster.id}"]`, $('#intelligenceFeed'));
      if (!feedItem) return;

      const category = DISASTER_CATEGORIES[disaster.category] || DISASTER_CATEGORIES.flood;
      const severity = SEVERITY_LEVELS[disaster.severity] || SEVERITY_LEVELS[1];

      feedItem.style.borderColor = category.color;
      feedItem.querySelector('.text-sm.font-medium').textContent = disaster.title;
      feedItem.querySelector('.text-xs.text-slate-300').textContent = disaster.description;
      feedItem.querySelector('.text-xs.text-slate-400').textContent = timeAgo(disaster.timestamp);
      feedItem.querySelector('.severity-badge').className = `severity-badge severity-${disaster.severity}`;
      feedItem.querySelector('.severity-badge').textContent = severity.name;
      feedItem.querySelector('.text-slate-400:not(.terminal-font)').textContent = disaster.location.country;
    }

    removeFeedItem(disasterId) {
      const feedItem = $(`[data-disaster-id="${disasterId}"]`, $('#intelligenceFeed'));
      if (feedItem) feedItem.remove();
    }

    renderAlertModal(disasterId = null) {
      const disasters = this.dataStore.getDisasters(this.filters);
      const content = $('#alertModalContent');
      
      if (disasterId) {
        const disaster = this.dataStore.disasters.find(t => t.id === disasterId);
        if (!disaster) return;
        
        const category = DISASTER_CATEGORIES[disaster.category] || DISASTER_CATEGORIES.flood;
        const severity = SEVERITY_LEVELS[disaster.severity] || SEVERITY_LEVELS[1];

        content.innerHTML = `
          <div class="space-y-4">
            <div class="flex items-center gap-3">
              <i class="${category.icon} text-2xl" style="color: ${category.color}"></i>
              <h3 class="text-lg font-semibold text-white">${disaster.title}</h3>
            </div>
            <div class="text-sm text-slate-300">${disaster.description}</div>
            <div class="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div class="text-xs text-slate-400 terminal-font">SEVERITY</div>
                <div class="severity-badge severity-${disaster.severity}">${severity.name}</div>
              </div>
              <div>
                <div class="text-xs text-slate-400 terminal-font">CONFIDENCE</div>
                <div class="text-white">${disaster.confidence}%</div>
              </div>
              <div>
                <div class="text-xs text-slate-400 terminal-font">LOCATION</div>
                <div class="text-white">${disaster.location.country}, ${disaster.location.region}</div>
              </div>
              <div>
                <div class="text-xs text-slate-400 terminal-font">TIMESTAMP</div>
                <div class="text-white">${formatTime(disaster.timestamp)} UTC</div>
              </div>
              <div>
                <div class="text-xs text-slate-400 terminal-font">SOURCE</div>
                <div class="text-white">${disaster.source}</div>
              </div>
            </div>
            <div class="flex gap-3">
              <button class="btn-track bg-green-500/20 text-green-400 hover:bg-green-500/30 border border-green-500/30 px-4 py-2 rounded-lg flex-1" data-disaster-id="${disaster.id}">
                <i class="fas fa-crosshairs"></i> Track on Map
              </button>
              <button class="btn-analyze bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 border border-blue-500/30 px-4 py-2 rounded-lg flex-1">
                <i class="fas fa-chart-line"></i> Analyze
              </button>
            </div>
          </div>
        `;

        content.querySelector('.btn-track').addEventListener('click', () => {
          this.mapController.focusOnDisaster(disaster.id);
          $('#alertModal').classList.remove('active');
          this.notificationSystem.show({
            title: 'Tracking Enabled',
            body: `Now tracking disaster ID: ${disaster.id.substring(0, 8)}`
          }, 'success');
        });

        content.querySelector('.btn-analyze').addEventListener('click', () => {
          this.notificationSystem.show({
            title: 'Analysis Initiated',
            body: `Detailed analysis for ${disaster.title} has been queued.`
          }, 'info');
        });
      } else {
        content.innerHTML = `
          <div class="space-y-4">
            <h3 class="text-lg font-semibold text-white">Active Disaster Alerts</h3>
            ${disasters.map(t => {
              const category = DISASTER_CATEGORIES[t.category] || DISASTER_CATEGORIES.flood;
              const severity = SEVERITY_LEVELS[t.severity] || SEVERITY_LEVELS[1];
              return `
                <div class="glass-panel rounded-lg p-4 border-l-4" style="border-color: ${category.color}">
                  <div class="flex items-start gap-3">
                    <i class="${category.icon}" style="color: ${category.color}"></i>
                    <div class="flex-1">
                      <div class="flex items-center justify-between">
                        <div class="text-sm font-medium text-white">${t.title}</div>
                        <div class="text-xs text-slate-400 terminal-font">${timeAgo(t.timestamp)}</div>
                      </div>
                      <div class="text-xs text-slate-300 mt-1">${t.description}</div>
                      <div class="flex items-center gap-2 mt-2 text-xs">
                        <span class="severity-badge severity-${t.severity}">${severity.name}</span>
                        <span class="text-slate-400">${t.location.country}</span>
                      </div>
                    </div>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
        `;
      }
    }

    updateDashboard() {
      const disasters = this.dataStore.getDisasters(this.filters);
      
      // Update metrics
      $('#activeDisasterCount').textContent = disasters.length;
      $('#activeDisasterCount').parentElement.querySelector('.text-xs.text-slate-500').textContent = 
        `+${disasters.filter(t => t.timestamp > Date.now() - 3600000).length} in last hour`;
      $('#visibleCount').textContent = disasters.length;
      $('#totalCount').textContent = this.dataStore.disasters.length;
      $('#alertBadge').textContent = disasters.filter(t => t.severity >= 4).length;
      
      // Update charts
      this.analyticsController.updateCharts(disasters);

      // Update regional assessments
      const regions = {
        'United States': { active: 0, delta: 0, severity: 1 },
        'India': { active: 0, delta: 0, severity: 1 },
        'Brazil': { active: 0, delta: 0, severity: 1 },
        'Australia': { active: 0, delta: 0, severity: 1 }
      };

      disasters.forEach(disaster => {
        const country = disaster.location.country;
        if (regions[country]) {
          regions[country].active++;
          if (disaster.timestamp > Date.now() - 86400000) {
            regions[country].delta++;
          }
          regions[country].severity = Math.max(regions[country].severity, disaster.severity);
        }
      });

      $$('.glass-panel.rounded-lg.p-4.border').forEach(panel => {
        const country = panel.querySelector('h4').textContent;
        const data = regions[country] || { active: 0, delta: 0, severity: 1 };
        const statusIndicator = panel.querySelector('.status-indicator');
        const severityText = panel.querySelector('.text-xs.terminal-font');
        
        panel.querySelector('.text-red-400:not(.terminal-font), .text-amber-400:not(.terminal-font), .text-green-400:not(.terminal-font)').textContent = data.active;
        panel.querySelector('.text-red-400:last-child, .text-amber-400:last-child, .text-green-400:last-child').textContent = data.delta ? `+${data.delta}` : '0';
        
        statusIndicator.className = 'status-indicator';
        severityText.className = 'text-xs terminal-font';
        
        if (data.severity >= 4) {
          statusIndicator.classList.add('offline');
          severityText.classList.add('text-red-400');
          severityText.textContent = data.severity === 5 ? 'CRITICAL' : 'HIGH';
          panel.classList.remove('border-amber-500/30', 'border-green-500/30');
          panel.classList.add('border-red-500/30');
        } else if (data.severity === 3) {
          statusIndicator.classList.add('maintenance');
          severityText.classList.add('text-amber-400');
          severityText.textContent = 'MEDIUM';
          panel.classList.remove('border-red-500/30', 'border-green-500/30');
          panel.classList.add('border-amber-500/30');
        } else {
          statusIndicator.classList.add('online');
          severityText.classList.add('text-green-400');
          severityText.textContent = 'LOW';
          panel.classList.remove('border-red-500/30', 'border-amber-500/30');
          panel.classList.add('border-green-500/30');
        }
      });
    }
  }

  // Initialize the application
  const app = new ClimateDisasterRisk();

  // Initial sample data
  app.generateSampleData();

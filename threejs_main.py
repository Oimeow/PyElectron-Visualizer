# -- Imports --
import numpy as np
from scipy.special import sph_harm_y, genlaguerre, factorial
import streamlit as st
import streamlit.components.v1 as components
import json


# -- WebApp Setup --
st.set_page_config(layout='wide')
st.title("Orbital Visualizer")
a0 = 1.0  # Bohr's constant (atomic radius)


# -- Precompute factorials --
FACTORIAL_CACHE = [factorial(i) for i in range(30)]


# -- Radial Wavefunction --
def R_nl(r, n, l):
    """
    Radial component of hydrogen atom wavefunction
    
    :param r: radial distance from nucleus
    :param n: principal energy level integer (quantum number)
    :param l: azimuthal/angular momentum integer describing shape of orbitals (quantum number)
    :return: radial wavefunction value at distance r
    """
    rho = 2*r/(n*a0)
    prefactor = np.sqrt(
        (2/(n*a0))**3 *
        FACTORIAL_CACHE[n-l-1] /
        (2 * n * FACTORIAL_CACHE[n+l])
    )
    laguerre = genlaguerre(n-l-1, 2*l+1)(rho)
    return prefactor * np.exp(-rho/2) * rho**l * laguerre


def sample_r_optimized(n, l, N):
    """
    Optimized rejection sampling for radial distance
    Uses larger batches and vectorized operations
    
    :param n: principal energy level integer (quantum number)
    :param l: azimuthal/angular momentum integer describing shape of orbitals (quantum number)
    :param N: Sample size (e.g., 50_000)
    :return: array of N radial distance samples
    """
    r_max = 20*n
    r_samples = []
    batch_size = max(N * 5, 10000)
    
    while len(r_samples) < N:
        r_trial = np.random.uniform(0, r_max, batch_size)
        R_val = R_nl(r_trial, n, l)
        prob = r_trial**2 * np.abs(R_val)**2
        p_max = prob.max()
        if p_max > 0:
            accept_prob = np.random.uniform(0, 1, batch_size)
            mask = accept_prob < (prob / p_max)
            r_samples.extend(r_trial[mask])
    
    return np.array(r_samples[:N])


def sample_angles_optimized(l, m, N):
    """
    Optimized rejection sampling for angular coordinates
    Uses spherical harmonics for probability distribution
    
    :param l: azimuthal/angular momentum integer describing shape of orbitals (quantum number)
    :param m: magnetic quantum number describing orientation of orbital in space (quantum number)
    :param N: Sample size (e.g., 50_000)
    :return: tuple of (theta, phi) arrays with N samples each
    """
    th_samples = []
    phi_samples = []
    
    th_test = np.linspace(0, np.pi, 100)
    phi_test = np.linspace(0, 2*np.pi, 100)
    TH, PH = np.meshgrid(th_test, phi_test)
    p_max = (np.abs(sph_harm_y(l, m, TH, PH))**2).max()
    
    batch_size = max(N * 5, 10000)
    
    while len(th_samples) < N:
        cos_th = np.random.uniform(-1, 1, batch_size)
        th = np.arccos(cos_th)
        phi = np.random.uniform(0, 2*np.pi, batch_size)
        prob = np.abs(sph_harm_y(l, m, th, phi))**2
        
        if p_max > 0:
            accept_prob = np.random.uniform(0, 1, batch_size)
            mask = accept_prob < (prob / p_max)
            th_samples.extend(th[mask])
            phi_samples.extend(phi[mask])
    
    return np.array(th_samples[:N]), np.array(phi_samples[:N])


@st.cache_data(show_spinner=False)
def sample_orbital_optimized(n, l, m, N):
    """
    Generate samples from hydrogen orbital wavefunction
    Combines radial and angular sampling with probability density calculation
    
    :param n: principal energy level integer (quantum number)
    :param l: azimuthal/angular momentum integer describing shape of orbitals (quantum number)
    :param m: magnetic quantum number describing orientation of orbital in space (quantum number)
    :param N: Sample size (e.g., 50_000)
    :return: tuple of (x, y, z, pdf) where pdf is probability density |ψ|²
    """
    r = sample_r_optimized(n, l, N)
    th, phi = sample_angles_optimized(l, m, N)
    
    Y = sph_harm_y(l, m, th, phi)
    R = R_nl(r, n, l)
    psi = R * Y
    pdf = np.abs(psi) ** 2
    
    sin_th = np.sin(th)
    x = r * sin_th * np.cos(phi)
    y = r * sin_th * np.sin(phi)
    z = r * np.cos(th)
    
    return x, y, z, pdf


def create_threejs_viewer(x, y, z, pdf, particle_size, opacity, colormap='viridis', 
                          camera_state=None, rotation_sensitivity=1.0, 
                          pan_sensitivity=1.0, zoom_sensitivity=1.0, color_scale=1.0):
    """
    Create Three.js HTML viewer with persistent camera state and 3D spheres
    
    :param x, y, z: Cartesian coordinates of orbital points
    :param pdf: Probability density values for each point
    :param particle_size: Size of rendered spheres
    :param opacity: Transparency of spheres (0-1)
    :param colormap: Name of colormap to use
    :param camera_state: Dictionary with camera position and rotation
    :param rotation_sensitivity: Mouse rotation speed multiplier
    :param pan_sensitivity: Mouse pan speed multiplier
    :param zoom_sensitivity: Mouse zoom speed multiplier
    :param color_scale: Power scaling for color mapping (higher = more saturated colors)
    :return: HTML string for Three.js viewer
    """
    
    # Calculate bounding box for scale reference
    max_extent = max(np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z)))
    box_size = max_extent * 1.1  # 10% larger than data
    
    # Normalize PDF for color mapping
    pdf_normalized = (pdf - pdf.min()) / (pdf.max() - pdf.min())
    
    # Apply color scaling (push values toward high end)
    pdf_scaled = np.power(pdf_normalized, 1.0 / color_scale)
    
    # Create positions array (interleaved x,y,z)
    positions = np.column_stack([x, y, z]).flatten().tolist()
    
    # Define colormap gradients (color stops from low to high)
    colormaps = {
        'viridis': [(0.267004, 0.004874, 0.329415), (0.282623, 0.140926, 0.457517), 
                    (0.253935, 0.265254, 0.529983), (0.206756, 0.371758, 0.553117),
                    (0.163625, 0.471133, 0.558148), (0.127568, 0.566949, 0.550556),
                    (0.134692, 0.658636, 0.517649), (0.266941, 0.748751, 0.440573),
                    (0.477504, 0.821444, 0.318195), (0.741388, 0.873449, 0.149561),
                    (0.993248, 0.906157, 0.143936)],
        'plasma': [(0.050383, 0.029803, 0.527975), (0.285282, 0.011105, 0.595428),
                   (0.476205, 0.011783, 0.619659), (0.647343, 0.111307, 0.570634),
                   (0.786283, 0.207078, 0.489221), (0.888941, 0.317654, 0.398063),
                   (0.957854, 0.447483, 0.326634), (0.987622, 0.594620, 0.290139),
                   (0.982894, 0.755750, 0.302830), (0.939173, 0.907731, 0.433714),
                   (0.940015, 0.975158, 0.131326)],
        'inferno': [(0.001462, 0.000466, 0.013866), (0.087411, 0.044556, 0.224813),
                    (0.258234, 0.038571, 0.406485), (0.416331, 0.090203, 0.432943),
                    (0.578304, 0.148039, 0.404411), (0.735683, 0.215906, 0.330245),
                    (0.865006, 0.316822, 0.226055), (0.952552, 0.457757, 0.131326),
                    (0.987622, 0.645320, 0.039886), (0.988362, 0.809365, 0.145357),
                    (0.988362, 0.998364, 0.644924)],
        'magma': [(0.001462, 0.000466, 0.013866), (0.081029, 0.041426, 0.220124),
                  (0.235739, 0.057873, 0.417331), (0.382914, 0.102815, 0.493666),
                  (0.520908, 0.166606, 0.513531), (0.658463, 0.240746, 0.476930),
                  (0.800215, 0.334051, 0.402597), (0.904281, 0.458512, 0.351413),
                  (0.967983, 0.611140, 0.427397), (0.994738, 0.780018, 0.618099),
                  (0.987053, 0.991438, 0.749504)],
        'turbo': [(0.18995, 0.07176, 0.23217), (0.13840, 0.25205, 0.58634),
                  (0.13211, 0.46452, 0.75466), (0.27596, 0.66014, 0.75151),
                  (0.53779, 0.79978, 0.58085), (0.78932, 0.85495, 0.34893),
                  (0.96696, 0.82256, 0.13046), (0.98447, 0.64624, 0.09766),
                  (0.88836, 0.45096, 0.13974), (0.72573, 0.27349, 0.17687),
                  (0.49602, 0.01960, 0.01893)],
        'hot': [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0),
                (1.0, 0.5, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)],
        'jet': [(0.0, 0.0, 0.5), (0.0, 0.0, 1.0), (0.0, 0.5, 1.0),
                (0.0, 1.0, 1.0), (0.5, 1.0, 0.5), (1.0, 1.0, 0.0),
                (1.0, 0.5, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
    }
    
    # Get gradient for selected colormap
    gradient = colormaps.get(colormap, colormaps['viridis'])
    
    # Create colors by interpolating through gradient
    colors = []
    for p in pdf_scaled:
        # Map p (0-1) to position in gradient
        idx_float = p * (len(gradient) - 1)
        idx_low = int(np.floor(idx_float))
        idx_high = min(idx_low + 1, len(gradient) - 1)
        t = idx_float - idx_low  # Interpolation factor
        
        # Linear interpolation between two gradient stops
        color_low = gradient[idx_low]
        color_high = gradient[idx_high]
        
        r = color_low[0] + t * (color_high[0] - color_low[0])
        g = color_low[1] + t * (color_high[1] - color_low[1])
        b = color_low[2] + t * (color_high[2] - color_low[2])
        
        colors.extend([r, g, b])
    
    # Default camera state if not provided
    if camera_state is None:
        camera_state = {
            'position': {'x': 0, 'y': 0, 'z': 20},
            'rotation': {'x': 0, 'y': 0, 'z': 0}
        }
    
    # Generate colorbar gradient CSS
    def get_colorbar_gradient(cmap):
        gradients = {
            'viridis': '#440154, #31688e, #35b779, #fde724',
            'plasma': '#0d0887, #7e03a8, #cc4778, #f89540, #f0f921',
            'inferno': '#000004, #420a68, #932667, #dd513a, #fca50a, #fcffa4',
            'magma': '#000004, #3b0f70, #8c2981, #de4968, #fe9f6d, #fcfdbf',
            'turbo': '#30123b, #4777ef, #1ac7c2, #a0fc3c, #faba39, #e8584a',
            'hot': '#000000, #ff0000, #ffff00, #ffffff',
            'jet': '#000080, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000, #800000'
        }
        return gradients.get(cmap, gradients['viridis'])
    
    colorbar_gradient = get_colorbar_gradient(colormap)
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; overflow: hidden; }}
            canvas {{ 
                display: block; 
                cursor: none; /* Hide default cursor */
            }}
            #cursor {{
                position: fixed;
                width: 16px;
                height: 16px;
                background: white;
                border-radius: 50%;
                pointer-events: none;
                z-index: 9999;
                mix-blend-mode: difference; /* Inverts colors underneath */
                transform: translate(-50%, -50%);
            }}
            #info {{
                position: absolute;
                top: 10px;
                left: 10px;
                color: white;
                font-family: monospace;
                background: rgba(0,0,0,0.5);
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
            }}
            #colorbar {{
                position: absolute;
                bottom: 30px;
                right: 30px;
                width: 30px;
                height: 250px;
                border: 2px solid rgba(255,255,255,0.3);
                border-radius: 5px;
                background: linear-gradient(to top, {colorbar_gradient});
            }}
            #colorbar-label {{
                position: absolute;
                bottom: 30px;
                right: 70px;
                color: white;
                font-family: monospace;
                font-size: 11px;
                text-align: right;
            }}
            .colorbar-tick {{
                margin: 5px 0;
                opacity: 0.8;
            }}
        </style>
    </head>
    <body>
        <div id="cursor"></div>
        <div id="colorbar"></div>
        <div id="colorbar-label">
            <div class="colorbar-tick">High</div>
            <div class="colorbar-tick" style="margin-top: 85px;">|ψ|²</div>
            <div class="colorbar-tick" style="margin-top: 85px;">Low</div>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
            // ===== CUSTOM CURSOR TRACKING =====
            const cursor = document.getElementById('cursor');
            document.addEventListener('mousemove', (e) => {{
                cursor.style.left = e.clientX + 'px';
                cursor.style.top = e.clientY + 'px';
            }});
            
            // Hide cursor when mouse leaves window
            document.addEventListener('mouseleave', () => {{
                cursor.style.display = 'none';
            }});
            document.addEventListener('mouseenter', () => {{
                cursor.style.display = 'block';
            }});
            
            // ===== SCENE SETUP =====
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0a0a);
            
            const camera = new THREE.PerspectiveCamera(
                75, 
                window.innerWidth / window.innerHeight, 
                0.1, 
                1000
            );
            
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);
            
            // ===== PARTICLE SYSTEM (3D Instanced Spheres) =====
            const positions = new Float32Array({json.dumps(positions)});
            const colors = new Float32Array({json.dumps(colors)});
            const numPoints = positions.length / 3;
            
            // Low-poly sphere geometry for performance
            const sphereGeometry = new THREE.SphereGeometry(1, 8, 6);
            
            // Create material WITHOUT vertexColors (we'll use setColorAt instead)
            const material = new THREE.MeshBasicMaterial({{
                transparent: true,
                opacity: {opacity}
            }});
            
            const instancedMesh = new THREE.InstancedMesh(sphereGeometry, material, numPoints);
            
            // Temp objects for matrix manipulation
            const dummy = new THREE.Object3D();
            const color = new THREE.Color();
            const particleScale = {particle_size * 0.08};
            
            // Set position, scale, and color for each sphere
            for (let i = 0; i < numPoints; i++) {{
                const i3 = i * 3;
                
                // Set position and scale
                dummy.position.set(positions[i3], positions[i3 + 1], positions[i3 + 2]);
                dummy.scale.set(particleScale, particleScale, particleScale);
                dummy.updateMatrix();
                instancedMesh.setMatrixAt(i, dummy.matrix);
                
                // Set color from our colormap
                color.setRGB(colors[i3], colors[i3 + 1], colors[i3 + 2]);
                instancedMesh.setColorAt(i, color);
            }}
            
            scene.add(instancedMesh);
            
            // ===== NUCLEUS (red sphere at origin) =====
            const nucleusGeometry = new THREE.SphereGeometry(0.3, 16, 16);
            const nucleusMaterial = new THREE.MeshBasicMaterial({{ color: 0xff3333 }});
            const nucleus = new THREE.Mesh(nucleusGeometry, nucleusMaterial);
            scene.add(nucleus);
            
            // ===== WIREFRAME BOX for scale reference =====
            const boxSize = {box_size};
            const boxGeometry = new THREE.BoxGeometry(boxSize * 2, boxSize * 2, boxSize * 2);
            const wireframeGeometry = new THREE.EdgesGeometry(boxGeometry);
            const wireframeMaterial = new THREE.LineBasicMaterial({{ 
                color: 0x555555,
                transparent: true,
                opacity: 0.25
            }});
            const wireframeBox = new THREE.LineSegments(wireframeGeometry, wireframeMaterial);
            scene.add(wireframeBox);
            
            // ===== RESTORE CAMERA STATE WITH PROPER PERSISTENCE =====
            // Check if we should reset (clear localStorage)
            const urlParams = new URLSearchParams(window.location.search);
            const shouldReset = urlParams.has('reset_camera');
            
            if (shouldReset) {{
                localStorage.removeItem('orbitalCameraState');
                console.log('Camera state reset - localStorage cleared');
            }}
            
            // Priority: localStorage > Streamlit state
            let savedCameraState = null;
            
            // First, try localStorage (persists across everything)
            const storedState = localStorage.getItem('orbitalCameraState');
            if (storedState && !shouldReset) {{
                try {{
                    savedCameraState = JSON.parse(storedState);
                    console.log('Loaded camera state from localStorage:', savedCameraState);
                }} catch (e) {{
                    console.log('Could not parse stored camera state');
                }}
            }}
            
            // Fallback to Streamlit state if no localStorage
            if (!savedCameraState) {{
                savedCameraState = {json.dumps(camera_state)};
                console.log('Using default camera state');
            }}
            
            // Apply camera state
            camera.position.set(
                savedCameraState.position.x,
                savedCameraState.position.y,
                savedCameraState.position.z
            );
            instancedMesh.rotation.set(
                savedCameraState.rotation.x,
                savedCameraState.rotation.y,
                savedCameraState.rotation.z
            );
            nucleus.rotation.set(
                savedCameraState.rotation.x,
                savedCameraState.rotation.y,
                savedCameraState.rotation.z
            );
            wireframeBox.rotation.set(
                savedCameraState.rotation.x,
                savedCameraState.rotation.y,
                savedCameraState.rotation.z
            );
            
            // ===== MOUSE CONTROLS WITH SMOOTH DAMPING =====
            let isDragging = false;
            let isPanning = false;
            let previousMousePosition = {{ x: 0, y: 0 }};
            
            // Velocity tracking for smooth deceleration
            let rotationVelocity = {{ x: 0, y: 0 }};
            let panVelocity = {{ x: 0, y: 0 }};
            const damping = 0.92; // Higher = less damping (0.9-0.95 is good)
            const acceleration = 0.3; // How quickly velocity builds
            
            renderer.domElement.addEventListener('mousedown', (e) => {{
                isDragging = true;
                isPanning = e.button === 2; // Right click
                previousMousePosition = {{ x: e.offsetX, y: e.offsetY }};
                
                // Stop all momentum when user grabs again
                rotationVelocity = {{ x: 0, y: 0 }};
                panVelocity = {{ x: 0, y: 0 }};
            }});
            
            renderer.domElement.addEventListener('mouseup', () => {{
                isDragging = false;
                isPanning = false;
                saveCameraState();
            }});
            
            renderer.domElement.addEventListener('mousemove', (e) => {{
                const deltaX = e.offsetX - previousMousePosition.x;
                const deltaY = e.offsetY - previousMousePosition.y;
                
                if (isDragging) {{
                    if (isPanning) {{
                        // Pan camera with smooth acceleration
                        const panSpeed = 0.02 * {pan_sensitivity};
                        const targetVelX = deltaX * panSpeed;
                        const targetVelY = deltaY * panSpeed;
                        
                        // Smooth acceleration toward target velocity
                        panVelocity.x += (targetVelX - panVelocity.x) * acceleration;
                        panVelocity.y += (targetVelY - panVelocity.y) * acceleration;
                        
                    }} else {{
                        // Rotate with smooth acceleration
                        const rotSpeed = 0.01 * {rotation_sensitivity};
                        const targetVelX = deltaY * rotSpeed;
                        const targetVelY = deltaX * rotSpeed;
                        
                        // Smooth acceleration toward target velocity
                        rotationVelocity.x += (targetVelX - rotationVelocity.x) * acceleration;
                        rotationVelocity.y += (targetVelY - rotationVelocity.y) * acceleration;
                    }}
                }}
                
                previousMousePosition = {{ x: e.offsetX, y: e.offsetY }};
            }});
            
            renderer.domElement.addEventListener('contextmenu', (e) => {{
                e.preventDefault();
            }});
            
            // Logarithmic zoom with mouse wheel and sensitivity
            renderer.domElement.addEventListener('wheel', (e) => {{
                e.preventDefault();
                
                // Logarithmic zoom: slower as you get closer
                const currentDist = Math.abs(camera.position.z);
                const logFactor = Math.log10(currentDist + 1) + 0.5;
                const zoomSpeed = 0.01 * {zoom_sensitivity};
                
                const delta = e.deltaY * zoomSpeed * logFactor;
                camera.position.z += delta;
                camera.position.z = Math.max(0.5, camera.position.z);
                
                saveCameraState();
            }});
            
            // ===== SAVE CAMERA STATE TO LOCALSTORAGE =====
            function saveCameraState() {{
                const state = {{
                    position: {{
                        x: camera.position.x,
                        y: camera.position.y,
                        z: camera.position.z
                    }},
                    rotation: {{
                        x: instancedMesh.rotation.x,
                        y: instancedMesh.rotation.y,
                        z: instancedMesh.rotation.z
                    }}
                }};
                
                // Store in localStorage for TRUE persistence
                try {{
                    localStorage.setItem('orbitalCameraState', JSON.stringify(state));
                }} catch (e) {{
                    console.error('Failed to save camera state:', e);
                }}
            }}
            
            // Handle window resize
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});
            
            // Animation loop with smooth damping
            let lastSaveTime = 0;
            const saveInterval = 100; // Save every 100ms instead of every frame
            
            function animate() {{
                requestAnimationFrame(animate);
                
                // Apply rotation velocity with damping
                if (Math.abs(rotationVelocity.x) > 0.0001 || Math.abs(rotationVelocity.y) > 0.0001) {{
                    instancedMesh.rotation.x += rotationVelocity.x;
                    instancedMesh.rotation.y += rotationVelocity.y;
                    nucleus.rotation.x += rotationVelocity.x;
                    nucleus.rotation.y += rotationVelocity.y;
                    wireframeBox.rotation.x += rotationVelocity.x;
                    wireframeBox.rotation.y += rotationVelocity.y;
                    
                    // Apply damping
                    rotationVelocity.x *= damping;
                    rotationVelocity.y *= damping;
                    
                    // Save periodically during momentum
                    const now = Date.now();
                    if (now - lastSaveTime > saveInterval) {{
                        saveCameraState();
                        lastSaveTime = now;
                    }}
                }}
                
                // Apply pan velocity with damping
                if (Math.abs(panVelocity.x) > 0.0001 || Math.abs(panVelocity.y) > 0.0001) {{
                    camera.position.x -= panVelocity.x;
                    camera.position.y += panVelocity.y;
                    
                    // Apply damping
                    panVelocity.x *= damping;
                    panVelocity.y *= damping;
                    
                    // Save periodically during momentum
                    const now = Date.now();
                    if (now - lastSaveTime > saveInterval) {{
                        saveCameraState();
                        lastSaveTime = now;
                    }}
                }}
                
                renderer.render(scene, camera);
            }}
            
            animate();
        </script>
    </body>
    </html>
    """
    
    return html_template


def main():
    # Initialize session state for camera persistence
    if 'camera_state' not in st.session_state:
        st.session_state.camera_state = {
            'position': {'x': 0, 'y': 0, 'z': 20},
            'rotation': {'x': 0, 'y': 0, 'z': 0}
        }
    
    # Initialize quantum number persistence
    if 'l_value' not in st.session_state:
        st.session_state.l_value = 1
    if 'm_value' not in st.session_state:
        st.session_state.m_value = 0
    
    # Compact sidebar CSS with better margins and SVG cursor
    st.markdown("""
    <style>
        /* Custom cursor using SVG - this WILL work */
        * {
            cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16"><circle cx="8" cy="8" r="8" fill="white" style="mix-blend-mode: difference"/></svg>') 8 8, auto !important;
        }
        
        /* Sidebar spacing adjustments */
        .stSlider > div > div > div {
            padding-top: 0.25rem !important;
            padding-bottom: 0.5rem !important;
        }
        .stSlider [data-baseweb="slider"] {
            margin-top: -5px;
        }
        section[data-testid="stSidebar"] .element-container {
            margin-bottom: 0.5rem;
        }
        section[data-testid="stSidebar"] h2 {
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
        }
        section[data-testid="stSidebar"] h3 {
            margin-top: 0.75rem;
            margin-bottom: 0.5rem;
            font-size: 0.95rem;
        }
        section[data-testid="stSidebar"] hr {
            margin-top: 0.75rem;
            margin-bottom: 0.75rem;
        }
        section[data-testid="stSidebar"] .stSelectbox {
            margin-bottom: 0.5rem;
        }
        section[data-testid="stSidebar"] .stNumberInput {
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar Controls
    st.sidebar.header("Quantum Numbers")
    n = st.sidebar.slider('n', 1, 10, 2, step=1, key='n_slider')
    
    # Handle l slider with persistence
    l_max = n - 1
    l_min = 0
    
    # Clamp stored l_value to valid range
    if st.session_state.l_value > l_max:
        st.session_state.l_value = l_max
    if st.session_state.l_value < l_min:
        st.session_state.l_value = l_min
    
    if l_max == 0:
        l = st.sidebar.number_input('l', min_value=0, max_value=0, value=0, step=1, key='l_input', disabled=True)
        st.session_state.l_value = 0
    else:
        l = st.sidebar.slider('l', l_min, l_max, st.session_state.l_value, step=1, key='l_slider')
        st.session_state.l_value = l
    
    # Handle m slider with persistence
    m_min = -l
    m_max = l
    
    # Clamp stored m_value to valid range
    if st.session_state.m_value > m_max:
        st.session_state.m_value = m_max
    if st.session_state.m_value < m_min:
        st.session_state.m_value = m_min
    
    if m_min == m_max:
        m = st.sidebar.number_input('m', min_value=m_min, max_value=m_max, value=0, step=1, key='m_input', disabled=True)
        st.session_state.m_value = 0
    else:
        m = st.sidebar.slider('m', m_min, m_max, st.session_state.m_value, step=1, key='m_slider')
        st.session_state.m_value = m
    
    N = st.sidebar.slider('N (samples)', 1_000, 200_000, 10_000, step=500, key='n_slider_sample')
    
    st.sidebar.divider()
    st.sidebar.subheader("Visuals")
    
    particle_size = st.sidebar.slider("Size", 0.1, 10.0, 1.5, step=0.1, key='particle_size')
    opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 1.0, step=0.05, key='opacity')
    probability_filter = st.sidebar.slider("Filter", 0.0, 0.99, 0.0, step=0.01, key='prob_filter')
    
    colormap = st.sidebar.selectbox("Colormap", 
                                     ['viridis', 'plasma', 'inferno', 'magma', 'turbo', 
                                      'hot', 'jet'], key='colormap')
    
    color_scale = st.sidebar.slider("Color Scale", 0.1, 5.0, 3.0, step=0.1, key='color_scale')
    
    st.sidebar.divider()
    st.sidebar.subheader("Controls")
    
    rotation_sensitivity = st.sidebar.slider("Rotation", 0.1, 5.0, 1.0, step=0.1, key='rot_sens')
    pan_sensitivity = st.sidebar.slider("Pan", 0.1, 5.0, 1.0, step=0.1, key='pan_sens')
    zoom_sensitivity = st.sidebar.slider("Zoom", 0.1, 5.0, 1.0, step=0.1, key='zoom_sens')
    
    # Enforce quantum rules
    if l >= n:
        l = n - 1
    if abs(m) > l:
        m = 0
    
    # Calculate orbital
    with st.spinner("Calculating orbital..."):
        x, y, z, pdf = sample_orbital_optimized(n, l, m, N)
    
    # Apply probability filter
    if probability_filter > 0:
        filter_value = np.percentile(pdf, probability_filter * 100)
        mask = pdf > filter_value
        x, y, z, pdf = x[mask], y[mask], z[mask], pdf[mask]
    
    # Display info
    orbital_names = ['s', 'p', 'd', 'f', 'g', 'h', 'i']
    max_extent = max(np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z)))
    st.info(f"{n}{orbital_names[l]} **Orbital** | (n,l,m) = ({n},{l},{m}) | {len(x):,}e | Extent: ~{max_extent:.2f} Bohr radii")
    
    # Create and display Three.js viewer with persistent camera state and sensitivities
    html_content = create_threejs_viewer(
        x, y, z, pdf, particle_size, opacity, colormap, 
        camera_state=st.session_state.camera_state,
        rotation_sensitivity=rotation_sensitivity,
        pan_sensitivity=pan_sensitivity,
        zoom_sensitivity=zoom_sensitivity,
        color_scale=color_scale
    )
    
    # Display with message handler for camera state updates
    components.html(html_content, height=700, scrolling=False)


if __name__ == "__main__":
    main()
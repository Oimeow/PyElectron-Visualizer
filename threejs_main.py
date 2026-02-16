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
@st.cache_data
def precompute_factorials(max_n=10):
    return {i: factorial(i) for i in range(max_n * 2 + 10)}

FACTORIAL_CACHE = precompute_factorials()


# -- Radial Wavefunction --
def R_nl(r, n, l):
    rho = 2*r/(n*a0)
    prefactor = np.sqrt(
        (2/(n*a0))**3 *
        FACTORIAL_CACHE[n-l-1] /
        (2 * n * FACTORIAL_CACHE[n+l])
    )
    laguerre = genlaguerre(n-l-1, 2*l+1)(rho)
    return prefactor * np.exp(-rho/2) * rho**l * laguerre


def sample_r_optimized(n, l, N):
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
    """Create Three.js HTML viewer with persistent camera state and 3D spheres"""
    
    # Calculate bounding box for scale reference
    max_extent = max(np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z)))
    box_size = max_extent * 1.1  # 10% larger than data
    
    # Normalize PDF for color mapping
    pdf_normalized = (pdf - pdf.min()) / (pdf.max() - pdf.min())
    
    # Apply color scaling (push values toward high end)
    pdf_scaled = np.power(pdf_normalized, 1.0 / color_scale)
    
    # Create positions array (interleaved x,y,z)
    positions = np.column_stack([x, y, z]).flatten().tolist()
    
    # Create colors based on PDF with expanded colormap options
    colors = []
    for p in pdf_scaled:
        if colormap == 'viridis':
            r = 0.267 + 0.005 * p + 0.322 * p**2
            g = 0.005 + 0.549 * p + 0.319 * p**2
            b = 0.329 + 0.569 * p - 0.535 * p**2
        elif colormap == 'plasma':
            r = 0.050 + 0.900 * p
            g = 0.030 + 0.870 * p**3
            b = 0.528 + 0.472 * (1-p)
        elif colormap == 'inferno':
            r = 0.001462 + 0.998538 * p
            g = 0.000466 + 0.998534 * p**4
            b = 0.013866 + 0.762 * p**2
        elif colormap == 'magma':
            r = 0.001462 + 0.995 * p**0.5
            g = 0.001466 + 0.92 * p**3
            b = 0.017422 + 0.55 * p
        elif colormap == 'turbo':
            r = 0.19 + 0.81 * np.sin(p * np.pi - 0.5)
            g = 0.25 + 0.75 * np.sin(p * np.pi)
            b = 0.95 - 0.55 * p
        elif colormap == 'hot':
            r = min(1.0, 2.5 * p)
            g = max(0.0, 2.5 * p - 1.0)
            b = max(0.0, 2.5 * p - 2.0)
        elif colormap == 'cool':
            r = p
            g = 1.0 - p
            b = 1.0
        elif colormap == 'rainbow':
            hue = p * 0.8  # 0 to 0.8 (red to violet)
            if hue < 0.2:
                r, g, b = 1.0, hue * 5, 0.0
            elif hue < 0.4:
                r, g, b = 1.0 - (hue - 0.2) * 5, 1.0, 0.0
            elif hue < 0.6:
                r, g, b = 0.0, 1.0, (hue - 0.4) * 5
            else:
                r, g, b = (hue - 0.6) * 5, 1.0 - (hue - 0.6) * 5, 1.0
        elif colormap == 'jet':
            if p < 0.25:
                r, g, b = 0.0, 0.0, 0.5 + 2.0 * p
            elif p < 0.5:
                r, g, b = 0.0, 4.0 * (p - 0.25), 1.0
            elif p < 0.75:
                r, g, b = 4.0 * (p - 0.5), 1.0, 1.0 - 4.0 * (p - 0.5)
            else:
                r, g, b = 1.0, 1.0 - 4.0 * (p - 0.75), 0.0
        else:  # default viridis
            r = 0.267 + 0.005 * p + 0.322 * p**2
            g = 0.005 + 0.549 * p + 0.319 * p**2
            b = 0.329 + 0.569 * p - 0.535 * p**2
        
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
            'cool': '#00ffff, #ff00ff',
            'rainbow': '#ff0000, #ffff00, #00ff00, #00ffff, #0000ff, #ff00ff',
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
        
        /* Reduce top margin of main page */
        .main .block-container {
            padding-top: 1rem !important;  /* Default is ~5rem, reduce to 1rem or 0.5rem */
        }
        
        /* Remove title margin if needed */
        h1 {
            margin-top: 0 !important;
            padding-top: 0 !important;
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
    n = st.sidebar.slider('n', 1, 7, 2, step=1, key='n_slider')
    
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
    
    particle_size = st.sidebar.slider("Size", 0.1, 10.0, 1.0, step=0.1, key='particle_size')
    opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.5, step=0.05, key='opacity')
    probability_filter = st.sidebar.slider("Filter", 0.0, 0.99, 0.0, step=0.01, key='prob_filter')
    
    colormap = st.sidebar.selectbox("Colormap", 
                                     ['viridis', 'plasma', 'inferno', 'magma', 'turbo', 
                                      'cool', 'hot', 'rainbow', 'jet'], key='colormap')
    
    color_scale = st.sidebar.slider("Color Scale", 0.1, 5.0, 1.0, step=0.1, key='color_scale')
    
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
    st.info(f"{n}{orbital_names[l]} | (n,l,m) = ({n},{l},{m}) | {len(x):,}e | Extent: ~{max_extent:.2f}a₀ (Bohr Radii)")
    
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
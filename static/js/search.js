document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.getElementById('search-form');
    const searchInput = document.getElementById('search-input');
    const resultsContainer = document.getElementById('results-container');
    const errorMessage = document.getElementById('error-message');

    searchForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const query = searchInput.value.trim();
        if (query) {
            performSearch(query);
        }
    });

    function performSearch(query) {
        errorMessage.textContent = '';
        resultsContainer.style.display = 'none';
        resultsContainer.innerHTML = '<p>Buscando...</p>';
        fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `query=${encodeURIComponent(query)}`
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || `HTTP error! status: ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            errorMessage.textContent = `Error al buscar: ${error.message}. Por favor, intente de nuevo.`;
            resultsContainer.style.display = 'none';
        });
    }

    function displayResults(results) {
        if (!results.length) {
            resultsContainer.innerHTML = '<p>No se encontraron resultados.</p>';
            resultsContainer.style.display = 'block';
            return;
        }

        resultsContainer.innerHTML = results.map(result => {
            const audioFeatures = result.audio_features || {};
            return `
                <div class="result-item">
                    <div class="section details-section">
                        <h2>${result.name}</h2>
                        <p><strong>Tipo:</strong> ${result.type}</p>
                        <p><strong>Popularidad:</strong> ${result.popularity}</p>
                        <p><strong>Fecha de lanzamiento:</strong> ${result.release_date}</p>
                        <p><strong>Duración:</strong> ${Math.floor(result.duration_ms / 60000)}:${('0' + Math.floor((result.duration_ms % 60000) / 1000)).slice(-2)}</p>
                        <p><strong>Explícito:</strong> ${result.explicit ? 'Sí' : 'No'}</p>
                    </div>

                    <div class="section artist-info">
                        <h3>Información del Artista:</h3>
                        <p><strong>Nombre:</strong> ${result.artist.name}</p>
                        <p><strong>Popularidad:</strong> ${result.artist.popularity}</p>
                        <p><strong>Géneros Musicales:</strong> ${result.artist.genres.join(', ') || 'No especificado'}</p>
                        <p><strong>Número de Seguidores:</strong> ${result.artist.followers}</p>
                    </div>

                    <div class="section album-info">
                        <h3>Álbum:</h3>
                        <p><strong>Nombre:</strong> ${result.album.name}</p>
                        <p><strong>Tipo:</strong> ${result.album.type}</p>
                        <p><strong>Fecha de lanzamiento:</strong> ${result.album.release_date}</p>
                    </div>

                    <div class="section collaborations">
                        <h3>Colaboraciones en esta canción:</h3>
                        <ul class="collaborations-list">
                            ${result.collaborations.map(collab => `<li>${collab}</li>`).join('')}
                        </ul>
                    </div>

                    <div class="section audio-features">
                        <h3>Características de Audio:</h3>
                        <p><strong>Acousticness:</strong> ${audioFeatures.acousticness || 'No disponible'}</p>
                        <p><strong>Danceability:</strong> ${audioFeatures.danceability || 'No disponible'}</p>
                        <p><strong>Energy:</strong> ${audioFeatures.energy || 'No disponible'}</p>
                        <p><strong>Instrumentalness:</strong> ${audioFeatures.instrumentalness || 'No disponible'}</p>
                        <p><strong>Liveness:</strong> ${audioFeatures.liveness || 'No disponible'}</p>
                        <p><strong>Valence:</strong> ${audioFeatures.valence || 'No disponible'}</p>
                        <p><strong>Tempo:</strong> ${audioFeatures.tempo || 'No disponible'}</p>
                        <p><strong>Speechiness:</strong> ${audioFeatures.speechiness || 'No disponible'}</p>
                    </div>

                    <div class="section playlists">
                        <h3>Playlists:</h3>
                        <ul class="playlists-list">
                            ${result.playlists && result.playlists.length ? 
                                result.playlists.map(playlist => `<li><a href="${playlist.external_urls}" target="_blank">${playlist.name}</a></li>`).join('') :
                                '<li>No hay playlists disponibles.</li>'
                            }
                        </ul>
                    </div>
                </div>
            `;
        }).join('');
        resultsContainer.style.display = 'block';
    }
});

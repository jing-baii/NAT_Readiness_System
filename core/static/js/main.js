// Utility Functions
function showAlert(message, type = 'success') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Find the first .container or .container-fluid
    const container = document.querySelector('.container, .container-fluid');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
    }
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.classList.remove('show');
        setTimeout(() => alertDiv.remove(), 150);
    }, 5000);
}

// Form Handling
function handleFormSubmit(form, submitButton, callback) {
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        const originalText = submitButton.innerHTML;
        submitButton.disabled = true;
        submitButton.innerHTML = `
            <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
            Loading...
        `;
        
        try {
            const formData = new FormData(form);
            const response = await fetch(form.action, {
                method: form.method,
                body: formData,
                headers: {
                    'X-CSRFToken': getCookie('csrftoken')
                }
            });
            
            const data = await response.json();
            
            if (response.ok) {
                if (callback) {
                    callback(data);
                }
                showAlert(data.message || 'Operation successful!', 'success');
            } else {
                showAlert(data.message || 'An error occurred. Please try again.', 'danger');
            }
        } catch (error) {
            console.error('Error:', error);
            showAlert('An unexpected error occurred. Please try again.', 'danger');
        } finally {
            // Restore button state
            submitButton.disabled = false;
            submitButton.innerHTML = originalText;
        }
    });
}

// CSRF Token
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// Dynamic Form Fields
function toggleFormFields(selectElement, fieldsToToggle) {
    const selectedValue = selectElement.value;
    Object.entries(fieldsToToggle).forEach(([value, elements]) => {
        elements.forEach(element => {
            if (Array.isArray(value)) {
                element.style.display = value.includes(selectedValue) ? 'block' : 'none';
            } else {
                element.style.display = value === selectedValue ? 'block' : 'none';
            }
        });
    });
}

// Focus Management
function setupFocusTrap(modal) {
    const focusableElements = modal.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstFocusable = focusableElements[0];
    const lastFocusable = focusableElements[focusableElements.length - 1];
    let previousActiveElement = null;

    function trapFocus(e) {
        if (e.key !== 'Tab') return;

        if (e.shiftKey) {
            if (document.activeElement === firstFocusable) {
                e.preventDefault();
                lastFocusable.focus();
            }
        } else {
            if (document.activeElement === lastFocusable) {
                e.preventDefault();
                firstFocusable.focus();
            }
        }
    }

    modal.addEventListener('keydown', trapFocus);

    // Store the previously focused element
    previousActiveElement = document.activeElement;

    // Focus the first element when modal opens
    firstFocusable?.focus();

    // Return cleanup function
    return () => {
        modal.removeEventListener('keydown', trapFocus);
        previousActiveElement?.focus();
    };
}

// Modal Cleanup Utility
function cleanupModal(modalId) {
    console.log('Cleaning up modal:', modalId);
    const modal = document.getElementById(modalId);
    if (!modal) {
        console.warn('Modal not found:', modalId);
        return;
    }

    // Remove any existing backdrop
    const backdrop = document.querySelector('.modal-backdrop');
    if (backdrop) {
        console.log('Removing existing backdrop');
        backdrop.remove();
    }

    // Remove modal-open class from body
    document.body.classList.remove('modal-open');
    document.body.style.overflow = '';
    document.body.style.paddingRight = '';

    // Reset modal state
    modal.classList.remove('show');
    modal.style.display = 'none';
    modal.setAttribute('aria-hidden', 'true');
    modal.removeAttribute('aria-modal');
    modal.removeAttribute('role');

    // Remove any existing event listeners
    const newModal = modal.cloneNode(true);
    modal.parentNode.replaceChild(newModal, modal);

    console.log('Modal cleanup completed');
    return newModal;
}

// Update setupModal function to use cleanup utility
function setupModal(modalId, formId, successCallback) {
    console.log('Setting up modal:', modalId, formId);
    try {
        // Clean up any existing modal instance
        const modal = cleanupModal(modalId);
        const form = document.getElementById(formId);
        const submitBtn = form?.querySelector('button[type="submit"]');
        
        console.log('Modal element:', modal);
        console.log('Form element:', form);
        console.log('Submit button:', submitBtn);
        
        if (!modal || !form) {
            console.error(`Modal setup failed: ${modalId} or ${formId} not found`);
            return;
        }

        // Initialize Bootstrap modal
        const bsModal = new bootstrap.Modal(modal);
        console.log('Bootstrap modal instance:', bsModal);
        
        // Setup focus trap
        let cleanupFocusTrap = null;
        
        modal.addEventListener('shown.bs.modal', () => {
            console.log('Modal shown event triggered');
            cleanupFocusTrap = setupFocusTrap(modal);
        });
        
        // Handle modal hidden event
        modal.addEventListener('hidden.bs.modal', () => {
            console.log('Modal hidden event triggered');
            if (form) form.reset();
            cleanupModal(modalId);
            // Cleanup focus trap
            if (cleanupFocusTrap) {
                console.log('Cleaning up focus trap');
                cleanupFocusTrap();
                cleanupFocusTrap = null;
            }
        });
        
        // Handle form submission
        if (form && submitBtn) {
            console.log('Setting up form submission handler');
            handleFormSubmit(form, submitBtn, (data) => {
                console.log('Form submitted successfully:', data);
                if (successCallback) {
                    successCallback(data);
                }
                bsModal.hide();
            });
        }

        return bsModal;
    } catch (error) {
        console.error('Error setting up modal:', error);
        return null;
    }
}

// File Upload Preview
function setupFilePreview(inputElement, previewElement) {
    inputElement.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                if (file.type.startsWith('image/')) {
                    previewElement.innerHTML = `<img src="${e.target.result}" class="img-fluid" alt="Preview">`;
                } else {
                    previewElement.innerHTML = `
                        <div class="p-3 border rounded">
                            <i class="fas fa-file me-2"></i>${file.name}
                        </div>
                    `;
                }
            };
            reader.readAsDataURL(file);
        } else {
            previewElement.innerHTML = '';
        }
    });
}

// Dropdown with Search
function setupSearchableDropdown(selectElement) {
    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.className = 'form-control mb-2';
    searchInput.placeholder = 'Search...';
    
    selectElement.parentNode.insertBefore(searchInput, selectElement);
    
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        Array.from(selectElement.options).forEach(option => {
            const text = option.text.toLowerCase();
            option.style.display = text.includes(searchTerm) ? '' : 'none';
        });
    });
}

// Table Sorting
function setupTableSorting(tableElement) {
    const headers = tableElement.querySelectorAll('th[data-sort]');
    headers.forEach(header => {
        header.style.cursor = 'pointer';
        header.addEventListener('click', () => {
            const column = header.dataset.sort;
            const isAsc = header.classList.contains('asc');
            
            // Remove sort classes from all headers
            headers.forEach(h => h.classList.remove('asc', 'desc'));
            
            // Add sort class to clicked header
            header.classList.add(isAsc ? 'desc' : 'asc');
            
            // Sort the table
            const tbody = tableElement.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            rows.sort((a, b) => {
                const aVal = a.querySelector(`td[data-${column}]`).dataset[column];
                const bVal = b.querySelector(`td[data-${column}]`).dataset[column];
                return isAsc ? bVal.localeCompare(aVal) : aVal.localeCompare(bVal);
            });
            
            rows.forEach(row => tbody.appendChild(row));
        });
    });
} 
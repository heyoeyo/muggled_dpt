/* Contains code for dynamically creating UI elements that can be more easily controlled in code */

// Helper used to hold css classes applied to UI DOM elements
const UICSS = {
    slider_container: "sliderui_container",
    toggle_container: "toggleui_container",
    toggle_button: "toggleui_btn",
    regular_button: "regularui_button",
}

// ....................................................................................................................

class SliderUI {

    /* Represents a single 'slider' with a value indicator */

    constructor(
        parent_elem,
        label,
        initial_value,
        min_value = 0,
        max_value = 100,
        step_value = 1,
        output_multiplier = 1,
    ) {

        // HTML slider element (main control component)
        const nice_label = label.toLowerCase().replaceAll(" ", "_");
        const slider_id = `${nice_label}_slider`;
        this.slider_elem = document.createElement("input");
        this.slider_elem.id = slider_id;
        this.slider_elem.type = "range";

        // Set initial slider state
        // IMPORTANT: Step has to be set first or things get weird!!!
        this.slider_elem.step = step_value;
        this.slider_elem.min = min_value;
        this.slider_elem.max = max_value;
        this.slider_elem.value = initial_value;
        this._min = min_value;
        this._max = max_value;
        this._initial_value = initial_value;

        // UI Label
        this.label_elem = document.createElement("label");
        this.label_elem.innerText = `${label}: `;
        this.label_elem.htmlFor = slider_id;

        // HTML element which shows current value (e.g. as text)
        this.out_elem = document.createElement("output");
        this.out_elem.for = slider_id;
        const mult_step = step_value * output_multiplier;
        this._out_decimals = mult_step >= 1 ? 0 : String(mult_step).split(".")[1].length;
        this._out_mult = output_multiplier;
        
        // HTML element which holds all other components and handles layout
        this.container = document.createElement("div");
        this.container.classList.add(UICSS.slider_container);
        this.container.appendChild(this.label_elem);
        this.container.appendChild(this.out_elem);
        this.container.appendChild(this.slider_elem);
        
        // Add UI to given parent element
        parent_elem.appendChild(this.container);
        
        // Initialize value/output
        this.value = parseFloat(this.slider_elem.value);
        this._update_output();

        // Add 'double click to reset' callbacks
        this.label_elem.addEventListener("dblclick", this.reset);
        this.out_elem.addEventListener("dblclick", this.reset);
    }

    trigger = () => this.slider_elem.dispatchEvent(new Event("input"));

    reset = () => this.set_value(this._initial_value);

    set_value = (new_value) => {
        this.slider_elem.value = Math.min(this._max, Math.max(this._min, new_value));
        this.trigger();
    }

    increment = (num_steps=1) => {
        this.set_value(parseFloat(this.value) + num_steps * parseFloat(this.slider_elem.step));
        return this;
    }

    decrement = (num_steps=1) => this.increment(-1 * num_steps);
    
    set_color = (red, green, blue) => {
        this.slider_elem.style.accentColor = `rgb(${red}, ${green}, ${blue})`;
    }

    _update_output = () => {
        this.value = parseFloat(this.slider_elem.value);
        this.out_elem.value = (this.value * this._out_mult).toFixed(this._out_decimals);
    }

    add_listener = (callback_func) => {
        this.slider_elem.addEventListener("input", (evt) => {
            this._update_output();
            callback_func(evt);
        })
        return this;
    }

    apply_css = (css_class_name, add_style=true) => {
        if (add_style) {
            this.container.classList.add(css_class_name);
        } else {
            this.container.classList.remove(css_class_name);
        }
        return this;
    }

    static set_colors = (rgb_value, ...slider_refs) => {
        slider_refs.forEach(elem => elem.set_color(...rgb_value));
    }
}

// ....................................................................................................................

class ToggleUI {

    /* Represents a single toggle button/value. Basically a fancy checkbox element */

    constructor(parent_elem, label, initial_state = false) {

        const nice_label = label.toLowerCase().replaceAll(" ", "_");
        const cbox_id = `${nice_label}_checkbox`;
        this.cbox_elem = document.createElement("input");
        this.cbox_elem.id = cbox_id;
        this.cbox_elem.type = "checkbox";
        this.cbox_elem.checked = initial_state;

        this.label_elem = document.createElement("label");
        this.label_elem.innerText = `${label}: `;
        this.label_elem.htmlFor = cbox_id;

        this.togbtn_elem = document.createElement("label");
        this.togbtn_elem.classList.add(UICSS.toggle_button);
        this.tslider_elem = document.createElement("div");
        this.togbtn_elem.appendChild(this.cbox_elem);
        this.togbtn_elem.appendChild(this.tslider_elem);
        
        this.container = document.createElement("div");
        this.container.classList.add(UICSS.toggle_container);
        this.container.appendChild(this.label_elem);
        this.container.appendChild(this.togbtn_elem);
        
        // Add UI to given parent element
        parent_elem.appendChild(this.container);
        
        // Initialize value/output
        this.set_color(200,200,200);
        this.value = this.cbox_elem.checked;
    }

    toggle = () => this.set_state(!this.cbox_elem.checked);
    
    trigger = () => this.cbox_elem.click();
    
    set_state = (new_state) => {
        if (new_state != this.cbox_elem.checked) this.trigger();
        return this;
    }
    
    set_color = (red, green, blue) => {
        this.togbtn_elem.style.setProperty("--accent_color_from_js", `rgb(${red}, ${green}, ${blue})`);
        return this;
    }

    add_listener = (callback_func) => {
        this.cbox_elem.addEventListener("change", (evt) => {
            this.value = this.cbox_elem.checked;
            callback_func(evt);
        });
        return this;
    }
}

// ....................................................................................................................

class ButtonUI {

    /* Represents a single 'regular' button, basically a clickable div. But usable like other UI classes */

    constructor(parent_elem, label) {

        const nice_label = label.toLowerCase().replaceAll(" ", "_");
        const btn_id = `${nice_label}_uibtn`;
        this.btn_elem = document.createElement("div");
        this.btn_elem.innerText = label;
        this.btn_elem.id = btn_id;
        this.btn_elem.classList.add(UICSS.regular_button);

        // Add UI to given parent element
        parent_elem.appendChild(this.btn_elem);

        // Initialize value/output
        this.set_color(200,200,200);
    }

    trigger = () => this.btn_elem.click();

    set_color = (red, green, blue) => {
        this.btn_elem.style.setProperty("--accent_color_from_js", `rgb(${red}, ${green}, ${blue})`);
        return this;
    }

    add_listener = (callback_func) => {
        this.btn_elem.addEventListener("click", (evt) => {
            callback_func(evt);
        });
        return this;
    }
}

// ....................................................................................................................

class UIRateLimiter {

    /*
    Simple helper used to manage state for rate-limiting event listeners. This class wraps around
    existing dom elements, and then handles event listeners for the element.

    // Example usage:
    const dom_element = document.getElementById("my_input_to_rate_limit");
    const limiter = new UIRateLimiter(dom_element, max_updates_per_second = 4);
    limiter.addEventListener("input", () => console.log("EVENT!"));

    */

    constructor(dom_element, max_updates_per_second = 5) {
        this.elem = dom_element;
        this._timeout_ref = null;
        this._timeout_ms = 1000.0 / max_updates_per_second;
        this._next_evt = null;
    }

    addEventListener = (event_type, callback_func) => {
        this.elem.addEventListener(event_type, (evt) => {

            this._next_evt = evt;
            if (this._timeout_ref === null) {
                this._timeout_ref = setTimeout(() => {
                    callback_func(this._next_evt);
                    this._timeout_ref = null;
                }, this._timeout_ms);

            }
        });
    }
}

// --------------------------------------------------------------------------------------------------------------------

function make_01_slider(parent_elem, label, initial_value, step = 0.01, min=0) {
    return new SliderUI(parent_elem, label, initial_value, min, 1.0, step, output_multiplier=100);
}

function append_spacer(parent_elem, css_class = null) {
    const elem = document.createElement("hr");
    if (css_class != null) {
        elem.classList.add(css_class);
    }
    parent_elem.appendChild(elem);
}
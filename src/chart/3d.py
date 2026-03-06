    def plot(self, title: str = "ConceptMatrix 3D", save_html: str = None):
        """
        Interactive 3D scatter of all stored cells.
        - Scalar cells: color + size encode the value.
        - Coord-list cells: rendered as connected line segments.
        - Hover shows coordinates and stored value.
        Requires a 3-D matrix.
        """
        if len(self.shape) != 3:
            raise NotImplementedError("plot() only supports 3-D matrices.")
        if not self._matrix_storage:
            print("Nothing to plot: matrix is empty.")
            return

        scalar_keys, scalar_vals = [], []
        coord_list_items = []

        for k, v in self._matrix_storage.items():
            if isinstance(v, (int, float)):
                scalar_keys.append(k)
                scalar_vals.append(v)
            elif isinstance(v, list):
                coord_list_items.append((k, v))

        traces = []

        # --- Scalar points ---
        if scalar_keys:
            xs = [c[0] for c in scalar_keys]
            ys = [c[1] for c in scalar_keys]
            zs = [c[2] for c in scalar_keys]
            vmin, vmax = min(scalar_vals), max(scalar_vals)
            span = (vmax - vmin) or 1
            sizes = [8 + 16 * (v - vmin) / span for v in scalar_vals]
            hover = [f"x={x}, y={y}, z={z}<br>value={v:.4g}"
                     for x, y, z, v in zip(xs, ys, zs, scalar_vals)]
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                name="scalars",
                marker=dict(
                    size=sizes, color=scalar_vals,
                    colorscale="Viridis",
                    colorbar=dict(title="Value"),
                    opacity=0.85,
                    line=dict(width=0.5, color="white"),
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            ))

        # --- Coord-list cells: one colored trace per target coord ---
        PALETTE = ["cyan", "magenta", "lime", "orange", "red", "yellow", "white"]

        for origin, targets in coord_list_items:
            ox, oy, oz = origin
            for i, t in enumerate(targets):
                color = PALETTE[i % len(PALETTE)]
                traces.append(go.Scatter3d(
                    x=[ox, t[0]], y=[oy, t[1]], z=[oz, t[2]],
                    mode="lines+markers",
                    name=f"link@{origin}→{t}",
                    line=dict(color=color, width=2),
                    marker=dict(size=[4, 0], color=color, symbol=["circle", "circle"]),
                    text=[f"origin={origin}", f"target={t}"],
                    hovertemplate="%{text}<extra></extra>",
                ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(text=title, x=0.5),
            scene=dict(
                xaxis=dict(title="X", range=[0, self.shape[0]]),
                yaxis=dict(title="Y", range=[0, self.shape[1]]),
                zaxis=dict(title="Z", range=[0, self.shape[2]]),
                bgcolor="rgb(10,10,20)",
            ),
            paper_bgcolor="rgb(20,20,30)",
            font=dict(color="white"),
            margin=dict(l=0, r=0, b=0, t=40),
        )

        if save_html:
            fig.write_html(save_html)
            print(f"Saved to {save_html}")

        fig.show()
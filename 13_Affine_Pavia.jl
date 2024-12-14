### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 301d8a41-12e0-4216-b849-902831f18d45
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ b211ba2a-c099-4a96-92f5-8ca491aba4b8
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ c57b192d-0ade-43fc-bc06-712e39c5dc0e
html"""<style>
main {
    max-width: 66%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 94f81a36-6b51-4360-a042-ec92272bde3c
@bind Location Select(["Pavia", "PaviaUni",])

# ╔═╡ 5f50aee1-fd90-4960-8483-49016f78d7d5
filepath = joinpath(@__DIR__, "MAT Files", "$Location.mat")

# ╔═╡ b4fddf5f-8f06-4467-958c-29147e2f7ea0
gt_filepath = joinpath(@__DIR__, "GT Files", "$Location.mat")

# ╔═╡ 32a947be-8be2-458c-ad40-c3286abe73bd
CACHEDIR = joinpath(@__DIR__, "cache_files", "Aerial Datasets")

# ╔═╡ 1dd2ad99-c288-47a1-9544-760b814a00c4
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ c4c09599-9d92-4d66-9d37-3fb873c35a05
vars = matread(filepath)

# ╔═╡ 0d473c0f-0a7f-4887-88fb-6be80a774924
vars_gt = matread(gt_filepath)

# ╔═╡ a42eb205-5161-4566-ac10-ec81ba39ac39
loc_dict_keys = Dict(
	"Pavia" => ("pavia", "pavia_gt"),
	"PaviaUni" => ("paviaU", "paviaU_gt")
)

# ╔═╡ d6d3a5fb-301b-4666-9f5e-3b3e7db1826a
data_key, gt_key = loc_dict_keys[Location]

# ╔═╡ 5da01022-9554-4861-a1c6-fce7265aec35
data = vars[data_key]

# ╔═╡ 0b4ebbac-3ade-4c19-8d4b-e325b90384e1
gt_data = vars_gt[gt_key]

# ╔═╡ c2dc6324-b478-4495-8ed1-7b082f0c6076
gt_labels = sort(unique(gt_data))

# ╔═╡ 93c009d6-6181-4e71-a365-d0f60bef2e0d
bg_indices = findall(gt_data .== 0)

# ╔═╡ 68c604fd-db76-4cc1-9f9c-fca4ad062320
md"""
### Defining mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ b0f75c2c-8fb2-4a9c-8ef1-70da20415ab1
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ 247ba18b-7eb8-4fcc-933a-dd6d98795496
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ 50658bef-37b5-40a9-bfdd-c316dd7a35a8
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ 208eee55-f441-4b7e-a0c3-0fd20b261fe0
with_theme() do
	fig = Figure(; size=(600, 600))
	labels = length(unique(gt_data))
	colors = Makie.Colors.distinguishable_colors(n_clusters+1)
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	ax1 = Axis(fig[1, 2], aspect=DataAspect(), yreversed=true)
	image!(ax, permutedims(data[:, :, band]))
	hm = heatmap!(ax1, permutedims(gt_data); colormap=Makie.Categorical(colors))
	fig
end

# ╔═╡ b7f7ee18-7abd-4a68-8076-75745d36f8b6
function affine_approx(X, k)
	bhat = mean(eachcol(X))
	Uhat = svd(X - bhat*ones(size(X,2))').U[:,1:k]
	return bhat, Uhat
end

# ╔═╡ 84b679c2-7a44-4d99-bd32-f02bab938795


# ╔═╡ Cell order:
# ╠═c57b192d-0ade-43fc-bc06-712e39c5dc0e
# ╠═301d8a41-12e0-4216-b849-902831f18d45
# ╠═b211ba2a-c099-4a96-92f5-8ca491aba4b8
# ╠═94f81a36-6b51-4360-a042-ec92272bde3c
# ╠═5f50aee1-fd90-4960-8483-49016f78d7d5
# ╠═b4fddf5f-8f06-4467-958c-29147e2f7ea0
# ╠═32a947be-8be2-458c-ad40-c3286abe73bd
# ╠═1dd2ad99-c288-47a1-9544-760b814a00c4
# ╠═c4c09599-9d92-4d66-9d37-3fb873c35a05
# ╠═0d473c0f-0a7f-4887-88fb-6be80a774924
# ╠═a42eb205-5161-4566-ac10-ec81ba39ac39
# ╠═d6d3a5fb-301b-4666-9f5e-3b3e7db1826a
# ╠═5da01022-9554-4861-a1c6-fce7265aec35
# ╠═0b4ebbac-3ade-4c19-8d4b-e325b90384e1
# ╠═c2dc6324-b478-4495-8ed1-7b082f0c6076
# ╠═93c009d6-6181-4e71-a365-d0f60bef2e0d
# ╟─68c604fd-db76-4cc1-9f9c-fca4ad062320
# ╠═b0f75c2c-8fb2-4a9c-8ef1-70da20415ab1
# ╠═247ba18b-7eb8-4fcc-933a-dd6d98795496
# ╠═50658bef-37b5-40a9-bfdd-c316dd7a35a8
# ╠═208eee55-f441-4b7e-a0c3-0fd20b261fe0
# ╠═b7f7ee18-7abd-4a68-8076-75745d36f8b6
# ╠═84b679c2-7a44-4d99-bd32-f02bab938795

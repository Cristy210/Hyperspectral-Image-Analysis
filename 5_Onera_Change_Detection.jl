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

# ╔═╡ e986b3f8-914b-47c8-94aa-0635330d695c
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 9531e81f-b0e4-49fc-8946-f6c9ffa04a82
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT, ArchGDAL

# ╔═╡ 60918ff2-8f51-11ef-3a74-2def7575fac8
html"""<style>
main {
    max-width: 66%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 9abb2611-c08b-4307-8cc2-911302598f12
@bind City Select(["abudhabi", "mumbai", "aguasclaras", "beihai", "beirut", "bercy", "bordeaux", "cupertino", "hongkong", "nantes", "paris", "pisa", "rennes", "saclay_e"])

# ╔═╡ 383efc0f-b775-4851-8a55-0ede19c78239
Path = Dict("_T1" => joinpath(@__DIR__, "Onera Satellite Change Detection dataset - Images", City, "imgs_1_rect"), "_T2" => joinpath(@__DIR__, "Onera Satellite Change Detection dataset - Images", City, "imgs_2_rect"))

# ╔═╡ 1606abd2-00f9-4bf6-ae8d-dfc9232e7ef7
FP_1 = Path["_T1"]

# ╔═╡ 8ea9100f-97f7-48b0-92a7-93de81fef45b
FP_2 = Path["_T2"]

# ╔═╡ b277bdcf-b5ad-4b7b-8537-5d323e3f05b6
Images_1 = glob("*.tif", FP_1)

# ╔═╡ 1befe06b-4060-45c5-a47b-70e206246a3e
Images_2 = glob("*.tif", FP_2)

# ╔═╡ d0fbbc1a-9355-4e4f-9ca4-3e823fc993cf
CACHEDIR = joinpath(@__DIR__, "cache_files", "Onera")

# ╔═╡ 613ed99b-e1a3-4a47-af25-c18ed6c074af
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 88559637-55db-4a20-87be-055a1660070e
Array_1 = [ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end for tif_file in Images_1]

# ╔═╡ f327d73f-c6ab-4832-901d-41c8708fb423
Array_2 = [ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end for tif_file in Images_2]

# ╔═╡ 3c98b6d2-f598-49dd-bdeb-000b06a25083
data_1 = cat(Array_1..., dims=3);

# ╔═╡ 4e9b9724-218d-498b-a33c-12d49d5c8887
data_2 = cat(Array_2..., dims=3);

# ╔═╡ 85eab518-fdce-4ed5-b7db-e89d4a3c0239
@bind band PlutoUI.Slider(1:size(data_1, 3), show_value=true)

# ╔═╡ dcf4212b-422f-4d2a-b1f1-2e18416ab147
with_theme() do
	fig = Figure(; size=(600, 500))
	ax_1 = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	ax_2 = Axis(fig[1, 2], aspect=DataAspect(), yreversed=true)
	image!(ax_1, data_1[:, :, band])
	image!(ax_2, data_2[:, :, band])
	fig
end

# ╔═╡ 432bd598-d170-43cc-8519-d0f2758fb083
begin
function affinity(X::Matrix; max_nz=10, chunksize=isqrt(size(X,2)),
	func = c -> exp(-2*acos(clamp(c,-1,1))))

	# Compute normalized spectra (so that inner product = cosine of angle)
	X = mapslices(normalize, X; dims=1)

	# Find nonzero values (in chunks)
	C_buf = similar(X, size(X,2), chunksize)    # pairwise cosine buffer
	s_buf = Vector{Int}(undef, size(X,2))       # sorting buffer
	nz_list = @withprogress mapreduce(vcat, enumerate(Iterators.partition(1:size(X,2), chunksize))) do (chunk_idx, chunk)
		# Compute cosine angles (for chunk) and store in appropriate buffer
		C_chunk = length(chunk) == chunksize ? C_buf : similar(X, size(X,2), length(chunk))
		mul!(C_chunk, X', view(X, :, chunk))

		# Zero out all but `max_nz` largest values
		nzs = map(chunk, eachcol(C_chunk)) do col, c
			idx = partialsortperm!(s_buf, c, 1:max_nz; rev=true)
			collect(idx), fill(col, max_nz), func.(view(c,idx))
		end

		# Log progress and return
		@logprogress chunk_idx/cld(size(X,2),chunksize)
		return nzs
	end

	# Form and return sparse array
	rows = reduce(vcat, getindex.(nz_list, 1))
	cols = reduce(vcat, getindex.(nz_list, 2))
	vals = reduce(vcat, getindex.(nz_list, 3))
	return sparse([rows; cols],[cols; rows],[vals; vals])
end
affinity(cube::Array{<:Real,3}; kwargs...) =
	affinity(permutedims(reshape(cube, :, size(cube,3))); kwargs...)
end

# ╔═╡ 6f1b577a-6042-4fcc-a21b-0b351e829dfd
max_nz = 20

# ╔═╡ 7a59cab0-8ba4-4c6b-8627-696c22fe0218
A_1 = cachet(joinpath(CACHEDIR, "Affinity_$(City)_T1$max_nz.bson")) do
	affinity(data; max_nz)
end

# ╔═╡ 876cc0e8-2880-4591-b212-1eac3d2c0b9a
A_2 = cachet(joinpath(CACHEDIR, "Affinity_$(City)_T2$max_nz.bson")) do
	affinity(data; max_nz)
end

# ╔═╡ 48267064-d704-439f-bd9e-f2f620a902e3
function embedding(A, k; seed=0, kwargs...)
	# Set seed for reproducibility
	Random.seed!(seed)

	# Compute node degrees and form Laplacian
	d = vec(sum(A; dims=2))
	Dsqrinv = sqrt(inv(Diagonal(d)))
	L = Symmetric(I - (Dsqrinv * A) * Dsqrinv)

	# Compute eigenvectors
	decomp, history = partialschur(L; nev=k, which=:SR, kwargs...)
	@info history

	return mapslices(normalize, decomp.Q; dims=2)
end

# ╔═╡ 8b04f5d4-6266-41f6-bacd-a0b091e37f19
n_clusters = 6

# ╔═╡ e2baaf78-4621-4f7a-bdf8-d5975db35d82


# ╔═╡ 69646bb2-f708-43dc-b58f-84b1fd8438c2
function batchkmeans(X, k, args...; nruns=100, kwargs...)
	runs = @withprogress map(1:nruns) do idx
		# Run K-means
		Random.seed!(idx)  # set seed for reproducibility
		result = with_logger(NullLogger()) do
			kmeans(X, k, args...; kwargs...)
		end

		# Log progress and return result
		@logprogress idx/nruns
		return result
	end

	# Print how many converged
	nconverged = count(run -> run.converged, runs)
	@info "$nconverged/$nruns runs converged"

	# Return runs sorted best to worst
	return sort(runs; by=run->run.totalcost)
end

# ╔═╡ be35e249-a19b-41ea-b110-28b21a352b58
aligned_assignments(clusterings, baseperm=1:maximum(first(clusterings).assignments)) = map(clusterings) do clustering
	# New labels determined by simple heuristic that aims to align different clusterings
	thresh! = a -> (a[a .< 0.2*sum(a)] .= 0; a)
	alignments = [thresh!(a) for a in eachrow(counts(clusterings[1], clustering))]
	new_labels = sortperm(alignments[baseperm]; rev=true)

	# Return assignments with new labels
	return [new_labels[l] for l in clustering.assignments]
end

# ╔═╡ 66ab6df2-0eb7-4ccd-8577-8412dcf391eb
spec_clusterings_1 = cachet(joinpath(CACHEDIR, "spec-max_nz_T1$(City)-$max_nz-n_clusters-$n_clusters.bson")) do
	V_1 = embedding(A_1, n_clusters; restarts=2000)
	batchkmeans(permutedims(V_1), n_clusters; maxiter=1000)
end

# ╔═╡ f0ddfef4-0a2f-4b54-846e-e5def0e9c5ea


# ╔═╡ d1c8fb7a-ee78-4601-949f-4c26b668802b
spec_clusterings_2 = cachet(joinpath(CACHEDIR, "spec-max_nz_T2$(City)-$max_nz-n_clusters-$n_clusters.bson")) do
	V_2 = embedding(A_2, n_clusters; restarts=2000)
	batchkmeans(permutedims(V_2), n_clusters; maxiter=1000)
end

# ╔═╡ 0304876e-4c79-4743-aed5-8b2a7bd39c51
spec_aligned_1 = aligned_assignments(spec_clusterings_1)

# ╔═╡ 842b72e2-aa38-45c5-9ddd-d68e0dc23c02
spec_aligned_2 = aligned_assignments(spec_clusterings_2)

# ╔═╡ 5686b3ec-fa41-4790-b0a6-f24d44d30635
@bind spec_clustering_idx PlutoUI.Slider(1:length(spec_clusterings_1); show_value=true)

# ╔═╡ b6e51128-10cb-407e-8ec1-5f994d68494b
mat_2 = reshape(spec_aligned_2[spec_clustering_idx], size(data_1, 1), size(data_1, 2));

# ╔═╡ 3ef0eb99-8968-4ed1-8832-08d891e95adb
mat_1 = reshape(spec_aligned_1[spec_clustering_idx], size(data_1, 1), size(data_1, 2));

# ╔═╡ 3e909260-73a7-4529-bb97-5427cba6e2ad
with_theme() do
	fig = Figure(; size=(900, 600))
	ax_1 = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	ax_2 = Axis(fig[1, 2], aspect=DataAspect(), yreversed=true)
	colors = Makie.Colors.distinguishable_colors(n_clusters)
	hm_1 = heatmap!(ax_1, mat_1; colormap=Makie.Categorical(colors))
	hm_2 = heatmap!(ax_2, mat_2; colormap=Makie.Categorical(colors))
	fig
end

# ╔═╡ 4919cb03-443b-4782-a52f-df4c06c7dccd


# ╔═╡ Cell order:
# ╟─60918ff2-8f51-11ef-3a74-2def7575fac8
# ╠═e986b3f8-914b-47c8-94aa-0635330d695c
# ╠═9531e81f-b0e4-49fc-8946-f6c9ffa04a82
# ╠═9abb2611-c08b-4307-8cc2-911302598f12
# ╠═383efc0f-b775-4851-8a55-0ede19c78239
# ╠═1606abd2-00f9-4bf6-ae8d-dfc9232e7ef7
# ╠═8ea9100f-97f7-48b0-92a7-93de81fef45b
# ╠═b277bdcf-b5ad-4b7b-8537-5d323e3f05b6
# ╠═1befe06b-4060-45c5-a47b-70e206246a3e
# ╠═d0fbbc1a-9355-4e4f-9ca4-3e823fc993cf
# ╠═613ed99b-e1a3-4a47-af25-c18ed6c074af
# ╠═88559637-55db-4a20-87be-055a1660070e
# ╠═f327d73f-c6ab-4832-901d-41c8708fb423
# ╠═3c98b6d2-f598-49dd-bdeb-000b06a25083
# ╠═4e9b9724-218d-498b-a33c-12d49d5c8887
# ╠═85eab518-fdce-4ed5-b7db-e89d4a3c0239
# ╠═dcf4212b-422f-4d2a-b1f1-2e18416ab147
# ╠═432bd598-d170-43cc-8519-d0f2758fb083
# ╠═6f1b577a-6042-4fcc-a21b-0b351e829dfd
# ╠═7a59cab0-8ba4-4c6b-8627-696c22fe0218
# ╠═876cc0e8-2880-4591-b212-1eac3d2c0b9a
# ╠═48267064-d704-439f-bd9e-f2f620a902e3
# ╠═8b04f5d4-6266-41f6-bacd-a0b091e37f19
# ╠═e2baaf78-4621-4f7a-bdf8-d5975db35d82
# ╠═69646bb2-f708-43dc-b58f-84b1fd8438c2
# ╠═be35e249-a19b-41ea-b110-28b21a352b58
# ╠═66ab6df2-0eb7-4ccd-8577-8412dcf391eb
# ╠═f0ddfef4-0a2f-4b54-846e-e5def0e9c5ea
# ╠═d1c8fb7a-ee78-4601-949f-4c26b668802b
# ╠═0304876e-4c79-4743-aed5-8b2a7bd39c51
# ╠═842b72e2-aa38-45c5-9ddd-d68e0dc23c02
# ╠═5686b3ec-fa41-4790-b0a6-f24d44d30635
# ╠═b6e51128-10cb-407e-8ec1-5f994d68494b
# ╠═3ef0eb99-8968-4ed1-8832-08d891e95adb
# ╠═3e909260-73a7-4529-bb97-5427cba6e2ad
# ╠═4919cb03-443b-4782-a52f-df4c06c7dccd

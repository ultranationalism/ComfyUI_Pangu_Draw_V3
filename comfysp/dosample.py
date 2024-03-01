def PanGu_Do_Sample():
    version_dict = VERSION2SPECS.get("PanGu-SDXL-base-1.0")
    seed_everything(seed)
    W, H = SD_XL_BASE_RATIOS[sd_xl_base_ratios]
    C = version_dict["C"]
    F = version_dict["f"]
    is_legacy = version_dict["is_legacy"]

    prompts = []
    negative_prompts = [negative_prompt]

    prompts.append(prompt)
    negative_prompts = negative_prompts * len(prompts)

    size_list = HIGH_SOLUTION_BASE_SIZE_LIST #if args.high_solution else BASE_SIZE_LIST
    assert (W, H) in size_list, f"(W, H)=({W}, {H}) is not in SIZE_LIST:{str(size_list)}"
    target_size_as_ind = size_list.index((W, H))

    value_dict = {
        "prompt": prompts,
        "negative_prompt": negative_prompt,
        "orig_width": orig_width if orig_width else W,
        "orig_height": orig_height if orig_height else H,
        "target_width": target_width if target_width else W,
        "target_height": target_height if target_height else H,
        "crop_coords_top": max(crop_coords_top if crop_coords_top else 0, 0),
        "crop_coords_left": max(crop_coords_left if crop_coords_left else 0, 0),
        "aesthetic_score": aesthetic_score if aesthetic_score else 6.0,
        "negative_aesthetic_score": negative_aesthetic_score if negative_aesthetic_score else 2.5,
        "aesthetic_scale": aesthetic_scale if aesthetic_scale else 0.0,
        "anime_scale": anime_scale if anime_scale else 0.0,
        "photography_scale": photography_scale if photography_scale else 0.0,
        "target_size_as_ind": target_size_as_ind,
    }

    sampler, num_rows, num_cols = init_sampling(
    sampler="PanGuEulerEDMSampler",
    num_cols=num_cols,
    guider="PanGuVanillaCFG",
    guidance_scale=guidance_scale,
    discretization="LegacyDDPMDiscretization",
    steps=steps,
    stage2strength=None,
    enable_pangu=True,
    other_scale=get_other_scale(value_dict),
)
    num_samples = num_rows * num_cols
    print("Txt2Img Sampling")
    s_time = time.time()
    samples = low.pangu_do_sample(
        high,
        sampler,
        value_dict,
        num_samples,
        H,
        W,
        C,
        F,
        force_uc_zero_embeddings=["txt"] if not is_legacy else [],
        return_latents=True,
        filter=filter,
        amp_level=00,
    )
    print(f"Txt2Img sample step {sampler.num_steps}, time cost: {time.time() - s_time:.2f}s")

    return samples

---
title: "4× RTX PRO 6000 Blackwell on Water, and the One Card That Wouldn't Behave"
date: 2026-06-12
description: "Converting four RTX PRO 6000 Blackwell cards to waterblocks, finding a VRM choke loose on the workbench, and getting back to 41k tok/s."
author: "Sabareesh"
draft: false
summary: "Converting four RTX PRO 6000 Blackwell cards to waterblocks, finding a VRM choke loose on the workbench, and getting back to 41k tok/s."
cover:
  image: "/blog-images/wb-rig-4gpu.jpg"
  alt: "Four waterblocked RTX PRO 6000 Blackwell GPUs on an open test bench"
  relative: false
images:
  - "/blog-images/wb-rig-4gpu.jpg"
tags: ["hardware", "gpu", "blackwell", "watercooling", "rtx-pro-6000", "vllm", "debugging"]
categories:
  - Hardware
  - GPU
disableShare: true
---

This rig exists to **train models**, not serve them. Four RTX PRO 6000 Blackwell cards in one chassis at 600 W each is 2.4 kW of heat to evict, and training runs are hours-to-days long with every card pinned at full TDP. Air coolers can do it for an inference burst; they cannot do it for a multi-day training job — the fans get loud, the cards stack their exhaust into each other, and the first one to thermal-throttle stalls the whole synchronous step.

So we converted the cards to waterblocks. We did **one card first** as a pilot, ran it for about a week, and only after that did we touch the other three. That sequencing matters — it's why we have a story to tell. The pilot card failed, taught us a lesson, and the lesson is the reason the other three went on without incident.

This post is the short version: what we did, what broke, what we learned, and where we landed.

## The rig

- 4× RTX PRO 6000 Blackwell Workstation (GB202, 96 GB GDDR7, 600 W)
- Threadripper Pro 7995WX on WRX90
- 4× Bykski waterblocks (full-cover, GPU + VRM + memory front-side)
- Custom loop: single distro/reservoir, two pumps, distilled water, two Alphacool NexXxoS XT45 Full Copper 1260 mm Super Nova radiators (9× 140 mm fans each), four GPUs plumbed in parallel
- 2× 1500 W PSUs (3 kW total budget) to feed the ~2.4 kW sustained draw; AC circuit got upgraded mid-build after an earlier all-cards-down event under load

![One of the two Alphacool XT45 1260 mm radiators — 3×3 grid of 140 mm fans](/blog-images/wb-radiator-fans.jpg)

That's one radiator. There are two of them.

![Both XT45 1260 mm radiators standing in the loop](/blog-images/wb-radiators-pair.jpg)

Why this much radiator for a 2.4 kW load? Two reasons. First, training jobs run for days — there's no "let the heat soak the radiator and recover later." The loop has to dump 2.4 kW continuously, and a smaller rad would force the fans into the high-RPM range where they're loud. With 18× 140 mm of surface, the fans run quietly and the coolant Δt across the rads stays small. Second, sizing for headroom means a single fan failure or a clogged dust filter doesn't end the run.

The waterblocks themselves are straightforward: pull the stock cooler, clean the die, fresh paste on the GPU, thermal pads on memory and VRMs, torque the block down in a star pattern. The catch on these cards is the backplate — the memory packages on the back also need cooling, which means either pads against the case panel or small finned heatsinks bonded on with thermal adhesive.

## The pilot card

The first card went into the loop alone — the rest were still on air. The plan was: convert one, soak-test it for a week under real workloads, only then commit to the other three. That kept the failure surface small while we learned the card.

For about a week, the pilot ran clean. Training, inference, full load, no Xids. Then it started falling off the bus under load. It would idle fine, run short bursts fine, and then drop out partway into a sustained workload. The dmesg signature was always the same:

```
NVRM: Xid (PCI:0000:02:00): 79, pid='<unknown>', GPU has fallen off the bus.
NVRM: Xid (PCI:0000:02:00): 154, Node Reboot Required
```

Xid 79 by itself is a generic "GPU stopped responding" — it can be driver, PCIe link, power, or the card. The companion 154 plus the PCIe AER logs showed a DPC containment event: the root port killed the link because the card stopped acknowledging transactions. That narrowed it to the card or its power delivery, not software.

The painful part is that everything else looked normal. The card enumerated. It loaded the driver. It ran short workloads. It only failed after the VRMs had been driving real current for a while.

The temptation here is to chase software: try a different driver, a different vLLM build, swap CUDA versions, blame torch.compile. I tried some of that. None of it changed anything. The next step was to stop guessing and look at the card.

## Pulling the block

![PCB with the waterblock removed, one VRM pad empty](/blog-images/wb-bare-pcb.jpg)

This is the back side of the GPU with the block off. The big metal lid in the middle is the GB202 IHS. The black ring around it is the VRM — each of those small black squares marked **85N** is a power inductor (a choke). They sit between the VRM MOSFETs and the GPU core, smoothing the switched current that feeds the die.

A 600 W card has a lot of these chokes for a reason. They share the load. Lose one and the rest pick up its share, but the regulator's feedback loop gets unhappy and the current waveform gets noisy.

So I pulled the waterblock off the pilot card cold — straight from the rig to the workbench, peeled the thermal pads. The pads on the VRM area came up. So did one of the chokes.

![Empty pad close-up, with the missing part on the cloth above](/blog-images/wb-empty-pad.jpg)

In the upper-right cluster, one footprint is empty. Two bare solder lands. The component that should be bridging them came up with the pad.

## The part

![The detached choke on a beige cloth](/blog-images/wb-choke-found.jpg)

![Same part, 85N marking visible](/blog-images/wb-choke-85n.jpg)

About 3 mm on a side, marked **85N**, identical to the 23 still on the board. A healthy SMD joint does not release to thermal-pad adhesion — you have to apply real force to lift one of these chokes. That this one came up means the solder underneath was already cracked.

That cracked joint is the whole story. The card had passed initial bring-up and ran fine at light loads for a week. Once it had spent enough hours pulling 600 W, thermal cycling on a marginal joint widened the crack until the inductor lost reliable contact under transient current. With one inductor effectively out of the picture under load, the remaining chokes carried its share, the regulator's feedback loop got noisier, ripple climbed, one of the GPU's internal rails dipped out of spec, and the card aborted the PCIe link rather than corrupt data. That's the Xid 79 + DPC containment signature in a nutshell — and it only shows up under real, sustained load.

## The lesson we got from doing it cold

Pulling the block at room temperature is what made the cracked joint visible — but if the joint had been healthy, doing it cold could just as easily have *created* the same defect on a different chip. Thermal pads on factory GPU coolers are sticky. The bond between the pad and the SMD parts is real. When you peel a cold pad off a VRM area, the pad pulls upward on whatever it's stuck to, and that force concentrates on the small SMD components, not on the big inductor and capacitor packages. Marginal joints fail. Healthy joints may survive but are stressed.

The fix is to **warm the GPU to about 90 °C first**, then disassemble while the pads are soft. At 90 °C the silicone in the pad goes pliable; it lets go of the components instead of holding them and lifting. We did this by running a brief compute load on the card to bring it up to temperature, then powering off and immediately starting the teardown — no hot-air gun, no heat plate, just the card's own heat.

We learned this the hard way on the pilot. Every card we converted after it — all three — came apart warm and went back together without surprise. Zero further incidents during conversion.

## Putting it back

Resoldering a power inductor onto a multi-layer GPU PCB is real microsolder work — small pads, fine-pitch neighbors, multi-layer copper that sinks heat aggressively. I didn't try to do it myself. I walked into a **SmartFix** — a national chain of phone and electronics repair shops — in Las Vegas, handed the bare PCB to their microsolder tech, and watched him do it. **$40, about twenty minutes.** These shops do BGA reballs and 0201-pitch work on iPhone boards every day; a 3 mm power inductor with two big flat pads is a small job to them.

That's the part most people miss about a "dead" GPU. If the failure is a single discrete component coming off the board, you don't need an RMA, a hot-air rework station, or a $500 microscope. You need to find the nearest shop that lists "microsoldering" or "logic board repair" on their website. Phone-repair chains, indie cellphone shops, and console-repair places all qualify. The bring-it-in-and-wait economics are very different from the ship-it-back-to-the-manufacturer flow.

Back home, fresh paste and pads on the GPU, waterblock torqued back down, into the loop. With the pilot card validated, we converted the remaining three cards using the warmup-before-disassembly procedure. No further incidents.

Powered on. `nvidia-smi` showed all four cards. Ran the standard stress suite on the repaired card alone:

```
Inference (vLLM):   10,283 tok/s sustained over 26 rounds
Training (PyTorch): 5,389 tok/s, ~213 TFLOPS, peak 54 °C, 609 W
Xid events:         none
```

Then all four together, same suite, started simultaneously:

```
                inference        training         peak temp    power
GPU 0           10,393 tok/s     210.5 TFLOPS     57 °C        612 W
GPU 1 (fixed)   10,234 tok/s     212.5 TFLOPS     54 °C        609 W
GPU 2           10,311 tok/s     208.3 TFLOPS     58 °C        608 W
GPU 3           10,143 tok/s     206.2 TFLOPS     58 °C        607 W
--------------------------------------------------------------------
Aggregate       41,081 tok/s     837.6 TFLOPS     58 °C       2.44 kW
Xid events: 0
```

The repaired card runs the coolest of the four — fresh paste and pads. The other three are within 4% of each other on inference and within 3% on training, which is about as tight as fleet matching gets on stock silicon. The water loop holds every card at full boost indefinitely; on air, the same workload throttled the cards to the mid-80s °C and clocked them down.

## Takeaways

- **Warm the GPU to about 90 °C before pulling the thermal pads.** This is the single most important thing in this post. Cold pads are sticky enough to lift small SMD parts off the PCB. Warm pads release cleanly. Run a brief load to bring the card up to temperature, power off, immediately disassemble. We learned this the hard way on the pilot and applied it to the other three with zero incidents.
- **Convert one card first, soak-test it, then commit.** A week of real workloads on the pilot is what surfaced the cracked joint. If we had converted all four at once we would have had four blocks to pull and four times the failure surface. One-at-a-time scales the risk to your actual learning rate.
- **A card that ran fine for a week is not proof of healthy hardware.** Marginal SMD joints can pass initial bring-up and only fail after enough thermal cycles at full load. "It worked yesterday" is not load-bearing evidence.
- **Xid 79 + DPC containment, only under sustained load, only on one card, is a hardware signal.** Driver swaps, CUDA reinstalls, and inference-engine theories were dead ends I spent hours on. The failure pattern itself told the story — listen to it earlier.
- **You probably don't need to RMA, and you probably shouldn't solder it yourself.** A local phone-repair chain with a microsolder tech can put a 3 mm SMD part back on a GPU PCB in twenty minutes for the price of dinner. The skill exists in your city; you just have to look for it.

## What the rig is doing now

The primary workload is **training**: multi-day BF16 runs across all four cards at sustained 600 W each, roughly 840 TFLOPS aggregate. Air cooling can't hold that envelope — the cards throttle into the mid-80s °C, the slowest card gates the synchronous step, and effective TFLOPS sag over the course of a long run. On water, every card stays at full boost for the entire job and step times stay flat. That's the whole point of the conversion.

When the rig is idle between training jobs, it doubles as an inference endpoint: a DP=4 vLLM deployment of Qwen3.6-27B, one independent instance per GPU, nginx load balancer in front, exposed through a Cloudflare tunnel. At 1,024 concurrent in-flight requests with 5,120-token outputs it does just over **8,000 output tokens per second** sustained, balanced to within 0.7% across the four endpoints, KV cache at 99.6%, every card pulling its full 600 W. Same thermal envelope as a training step, different shape.

The 85N inductor is back where it belongs.

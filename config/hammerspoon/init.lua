
-- voice-router hotkey — two-press flow with live transcription overlay:
--   Press 1: say target name → shows "→ firmware" overlay
--   Press 2: say command → live text appears in overlay, routes on release
-- Uses warm listen daemon for instant mic activation

local voiceRouter = {}
voiceRouter.optDown = false
voiceRouter.stateFile = os.getenv("HOME") .. "/.local/share/voice-claude/daemon-state.json"
voiceRouter.resultsFile = os.getenv("HOME") .. "/.local/share/voice-claude/daemon-results.jsonl"
voiceRouter.pidFile = os.getenv("HOME") .. "/.local/share/voice-claude/listen-daemon.pid"
voiceRouter.pollTimer = nil
voiceRouter.routeQueue = {}
voiceRouter.routing = false
voiceRouter.lastResultLine = 0
voiceRouter.pendingCount = 0
voiceRouter.canvas = nil
voiceRouter.chunksFile = os.getenv("HOME") .. "/.local/share/voice-claude/daemon-chunks.jsonl"
voiceRouter.lastChunkLine = 0
voiceRouter.chunkTimer = nil

-- Two-press state
voiceRouter.pendingTarget = nil
voiceRouter.targetOverlay = nil

-- Live transcription state
voiceRouter.liveOverlay = nil
voiceRouter.liveText = ""  -- accumulated chunk text during recording

-- Icon overlay for recording indicator
local iconPath = os.getenv("HOME") .. "/Documents/projects/voice-claude/icon.png"

local function showMicIndicator()
    if voiceRouter.canvas then
        voiceRouter.canvas:delete()
        voiceRouter.canvas = nil
    end

    local screen = hs.screen.mainScreen()
    local frame = screen:fullFrame()
    local size = 96

    voiceRouter.canvas = hs.canvas.new({
        x = frame.x + frame.w / 2 - size / 2,
        y = frame.y + frame.h / 2 - size / 2,
        w = size,
        h = size,
    })

    local img = hs.image.imageFromPath(iconPath)
    if img then
        voiceRouter.canvas:appendElements({
            type = "image",
            image = img,
            frame = { x = "0%", y = "0%", w = "100%", h = "100%" },
            imageScaling = "shrinkToFit",
        })
        voiceRouter.canvas:level(hs.canvas.windowLevels.overlay)
        voiceRouter.canvas:behavior(hs.canvas.windowBehaviors.canJoinAllSpaces
            + hs.canvas.windowBehaviors.stationary)
        voiceRouter.canvas:show()
    else
        hs.alert.show("🎤", nil, nil, 60)
    end
end

local function hideMicIndicator()
    if voiceRouter.canvas then
        voiceRouter.canvas:delete()
        voiceRouter.canvas = nil
    end
    hs.alert.closeAll()
end

local function showTargetOverlay(targetName)
    if voiceRouter.targetOverlay then
        voiceRouter.targetOverlay:delete()
        voiceRouter.targetOverlay = nil
    end

    local screen = hs.screen.mainScreen()
    local frame = screen:fullFrame()
    local width = 300
    local height = 50

    voiceRouter.targetOverlay = hs.canvas.new({
        x = frame.x + frame.w / 2 - width / 2,
        y = frame.y + frame.h / 2 + 60,
        w = width,
        h = height,
    })

    voiceRouter.targetOverlay:appendElements(
        {
            type = "rectangle",
            action = "fill",
            fillColor = { red = 0.1, green = 0.1, blue = 0.1, alpha = 0.85 },
            roundedRectRadii = { xRadius = 10, yRadius = 10 },
        },
        {
            type = "text",
            text = "→ " .. targetName,
            textColor = { red = 0.3, green = 0.9, blue = 0.5, alpha = 1.0 },
            textSize = 22,
            textAlignment = "center",
            frame = { x = "5%", y = "15%", w = "90%", h = "70%" },
        }
    )
    voiceRouter.targetOverlay:level(hs.canvas.windowLevels.overlay)
    voiceRouter.targetOverlay:behavior(hs.canvas.windowBehaviors.canJoinAllSpaces
        + hs.canvas.windowBehaviors.stationary)
    voiceRouter.targetOverlay:show()
end

local function hideTargetOverlay()
    if voiceRouter.targetOverlay then
        voiceRouter.targetOverlay:delete()
        voiceRouter.targetOverlay = nil
    end
end

local function updateLiveOverlay(text)
    -- Show live transcription text below the mic icon
    if voiceRouter.liveOverlay then
        voiceRouter.liveOverlay:delete()
        voiceRouter.liveOverlay = nil
    end

    if not text or #text == 0 then return end

    local screen = hs.screen.mainScreen()
    local frame = screen:fullFrame()
    local width = 600
    -- Auto-height based on text length
    local lines = math.ceil(#text / 50) + 1
    local height = math.max(50, lines * 28)

    voiceRouter.liveOverlay = hs.canvas.new({
        x = frame.x + frame.w / 2 - width / 2,
        y = frame.y + frame.h / 2 + 60,
        w = width,
        h = height,
    })

    -- Show target prefix if we have one
    local displayText = text
    if voiceRouter.pendingTarget then
        displayText = "→ " .. voiceRouter.pendingTarget .. "\n" .. text
    end

    voiceRouter.liveOverlay:appendElements(
        {
            type = "rectangle",
            action = "fill",
            fillColor = { red = 0.05, green = 0.05, blue = 0.1, alpha = 0.9 },
            roundedRectRadii = { xRadius = 10, yRadius = 10 },
        },
        {
            type = "text",
            text = displayText,
            textColor = { red = 0.8, green = 0.85, blue = 0.9, alpha = 0.9 },
            textSize = 18,
            textAlignment = "left",
            frame = { x = "4%", y = "8%", w = "92%", h = "84%" },
        }
    )
    voiceRouter.liveOverlay:level(hs.canvas.windowLevels.overlay)
    voiceRouter.liveOverlay:behavior(hs.canvas.windowBehaviors.canJoinAllSpaces
        + hs.canvas.windowBehaviors.stationary)
    voiceRouter.liveOverlay:show()
end

local function hideLiveOverlay()
    if voiceRouter.liveOverlay then
        voiceRouter.liveOverlay:delete()
        voiceRouter.liveOverlay = nil
    end
end

local function clearPendingTarget()
    voiceRouter.pendingTarget = nil
    hideTargetOverlay()
end

local function getDaemonPid()
    local f = io.open(voiceRouter.pidFile, "r")
    if not f then return nil end
    local pid = f:read("*a"):match("^%s*(.-)%s*$")
    f:close()
    if pid and #pid > 0 then return pid end
    return nil
end

local function signalDaemon(sig)
    local pid = getDaemonPid()
    if pid then
        local _, status = hs.execute("kill -0 " .. pid .. " 2>/dev/null")
        if not status then return false end
        hs.execute("kill -" .. sig .. " " .. pid)
        return true
    end
    return false
end

local function resolveTarget(text)
    local env = {
        PATH = os.getenv("HOME") .. "/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
        HOME = os.getenv("HOME"),
    }

    local task = hs.task.new(
        os.getenv("HOME") .. "/.local/bin/voice-route",
        function(exitCode, stdout, stderr)
            local target = nil
            if exitCode == 0 and stdout then
                target = stdout:match("^%s*(.-)%s*$")
            end

            if target and #target > 0 then
                voiceRouter.pendingTarget = target
                showTargetOverlay(target)
                hs.timer.doAfter(3, function() hideTargetOverlay() end)
            else
                hs.alert.show("? no session found", nil, nil, 1.5)
            end
        end,
        {"--resolve", text}
    )
    task:setEnvironment(env)
    task:start()
end

local function routeToTarget(target, text)
    local env = {
        PATH = os.getenv("HOME") .. "/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
        HOME = os.getenv("HOME"),
    }

    hideLiveOverlay()
    hideTargetOverlay()

    local task = hs.task.new(
        os.getenv("HOME") .. "/.local/bin/voice-route",
        function(exitCode, stdout, stderr)
            voiceRouter.routing = false
        end,
        {"--target", target, "--text", text}
    )
    task:setEnvironment(env)
    task:start()
end

local function classifyText(text, stickyTarget)
    local env = {
        PATH = os.getenv("HOME") .. "/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
        HOME = os.getenv("HOME"),
    }

    local task = hs.task.new(
        os.getenv("HOME") .. "/.local/bin/voice-route",
        function(exitCode, stdout, stderr)
            if exitCode ~= 0 or not stdout then
                routeToTarget(voiceRouter.pendingTarget, text)
                return
            end

            local ok, result = pcall(hs.json.decode, stdout)
            if not ok or not result then
                routeToTarget(voiceRouter.pendingTarget, text)
                return
            end

            if result.action == "switch" and result.target then
                clearPendingTarget()
                voiceRouter.pendingTarget = result.target
                showTargetOverlay(result.target)
                hs.timer.doAfter(3, function() hideTargetOverlay() end)
                if result.text and #result.text > 0 then
                    routeToTarget(result.target, result.text)
                end
            elseif result.action == "self" and result.text then
                routeToTarget(voiceRouter.pendingTarget, result.text)
            else
                routeToTarget(voiceRouter.pendingTarget, text)
            end
        end,
        {"--classify", text, "--sticky-target", stickyTarget}
    )
    task:setEnvironment(env)
    task:start()
end

local function processQueue()
    if voiceRouter.routing or #voiceRouter.routeQueue == 0 then return end

    voiceRouter.routing = true
    local text = table.remove(voiceRouter.routeQueue, 1)

    if not text or #text == 0 then
        voiceRouter.routing = false
        processQueue()
        return
    end

    -- Tier 1: Fast-path for unambiguous focus verbs
    local lower = text:lower()
    if lower:match("^go%s+to%s") or lower:match("^switch%s+to%s") or lower:match("^focus%s") then
        voiceRouter.routing = false
        clearPendingTarget()
        resolveTarget(text)
        return
    end

    if voiceRouter.pendingTarget then
        classifyText(text, voiceRouter.pendingTarget)
    else
        -- No sticky target — use voice-route --text which handles
        -- target extraction and last-active fallback
        local env = {
            PATH = os.getenv("HOME") .. "/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
            HOME = os.getenv("HOME"),
        }
        local task = hs.task.new(
            os.getenv("HOME") .. "/.local/bin/voice-route",
            function(exitCode, stdout, stderr)
                voiceRouter.routing = false
                -- If the route resolved a target, capture it as sticky
                if exitCode == 0 and stdout then
                    local target = stdout:match("→%s*([^:]+)")
                    if target then
                        voiceRouter.pendingTarget = target:match("^%s*(.-)%s*$")
                    end
                end
            end,
            {"--text", text}
        )
        task:setEnvironment(env)
        task:start()
    end
end

local function consumeResults()
    local f = io.open(voiceRouter.resultsFile, "r")
    if not f then return end

    local lineNum = 0
    for line in f:lines() do
        lineNum = lineNum + 1
        if lineNum > voiceRouter.lastResultLine then
            local ok, result = pcall(hs.json.decode, line)
            if ok and result and result.text and #result.text > 0 then
                table.insert(voiceRouter.routeQueue, result.text)
                voiceRouter.pendingCount = math.max(0, voiceRouter.pendingCount - 1)
            elseif ok then
                voiceRouter.pendingCount = math.max(0, voiceRouter.pendingCount - 1)
            end
            voiceRouter.lastResultLine = lineNum
        end
    end
    f:close()

    processQueue()
end

-- Live transcription: show chunks in overlay as they arrive
local function consumeChunks()
    local f = io.open(voiceRouter.chunksFile, "r")
    if not f then return end

    local lineNum = 0
    for line in f:lines() do
        lineNum = lineNum + 1
        if lineNum > voiceRouter.lastChunkLine then
            local ok, chunk = pcall(hs.json.decode, line)
            if ok and chunk and chunk.text and #chunk.text > 0 then
                -- Append chunk text to live display
                if #voiceRouter.liveText > 0 then
                    voiceRouter.liveText = voiceRouter.liveText .. " " .. chunk.text
                else
                    voiceRouter.liveText = chunk.text
                end
                updateLiveOverlay(voiceRouter.liveText)
            end
            voiceRouter.lastChunkLine = lineNum
        end
    end
    f:close()
end

local function startChunkPolling()
    if voiceRouter.chunkTimer then return end
    voiceRouter.lastChunkLine = 0
    voiceRouter.liveText = ""
    voiceRouter.chunkTimer = hs.timer.doEvery(0.15, function()
        consumeChunks()

        if not voiceRouter.optDown then
            if voiceRouter.chunkTimer then
                voiceRouter.chunkTimer:stop()
                voiceRouter.chunkTimer = nil
            end
        end
    end)
end

local function ensurePolling()
    if voiceRouter.pollTimer then return end
    voiceRouter.pollTimer = hs.timer.doEvery(0.15, function()
        consumeResults()

        if voiceRouter.pendingCount <= 0 and #voiceRouter.routeQueue == 0
           and not voiceRouter.routing and not voiceRouter.optDown then
            local f = io.open(voiceRouter.resultsFile, "w")
            if f then f:close() end
            voiceRouter.lastResultLine = 0
            voiceRouter.pollTimer:stop()
            voiceRouter.pollTimer = nil
        end
    end)
end

voiceRouter.tap = hs.eventtap.new({hs.eventtap.event.types.flagsChanged}, function(event)
    local flags = event:getRawEventData().CGEventData.flags
    local rightOpt = (flags & 0x40) ~= 0 and (flags & 0x20) == 0
    local ctrlPressed = (flags & 0x40000) ~= 0
    local cmdPressed = (flags & 0x100000) ~= 0
    local shiftPressed = (flags & 0x20000) ~= 0

    local optOnly = rightOpt and not ctrlPressed and not cmdPressed and not shiftPressed

    if optOnly and not voiceRouter.optDown then
        voiceRouter.optDown = true
        signalDaemon("USR1")
        showMicIndicator()
        -- Start live transcription polling
        startChunkPolling()

    elseif not rightOpt and voiceRouter.optDown then
        voiceRouter.optDown = false
        signalDaemon("USR2")
        hideMicIndicator()
        -- Keep chunk polling for 3s after release to catch the final chunk
        -- (the chunk timer's stop condition checks optDown, so override it)
        if voiceRouter.chunkTimer then
            voiceRouter.chunkTimer:stop()
            voiceRouter.chunkTimer = nil
        end
        -- Poll rapidly for the final chunk
        local finalPollCount = 0
        voiceRouter.chunkTimer = hs.timer.doEvery(0.1, function()
            consumeChunks()
            finalPollCount = finalPollCount + 1
            if finalPollCount >= 30 then  -- 3 seconds max
                voiceRouter.chunkTimer:stop()
                voiceRouter.chunkTimer = nil
                -- Hide overlay after showing final text
                hs.timer.doAfter(2, function()
                    hideLiveOverlay()
                    voiceRouter.liveText = ""
                end)
            end
        end)
        voiceRouter.pendingCount = voiceRouter.pendingCount + 1
        ensurePolling()
    end

    return false
end)

voiceRouter.tap:start()

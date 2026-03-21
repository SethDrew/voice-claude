
-- voice-router hotkey — two-press flow:
--   Press 1: say target name → shows "→ firmware" overlay
--   Press 2: say command → routes to confirmed target
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

-- Two-press state
voiceRouter.pendingTarget = nil      -- confirmed target from first press
voiceRouter.targetTimeout = nil      -- timer to clear pending target
voiceRouter.targetOverlay = nil      -- canvas showing target name

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
    -- Clear any existing target overlay
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
        y = frame.y + frame.h / 2 + 60,  -- below the mic icon
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

local function clearPendingTarget()
    voiceRouter.pendingTarget = nil
    hideTargetOverlay()
    if voiceRouter.targetTimeout then
        voiceRouter.targetTimeout:stop()
        voiceRouter.targetTimeout = nil
    end
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
    -- Call voice-route --resolve to get the target without routing
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

                -- Show overlay for 3 seconds then hide (target stays sticky in memory)
                hs.timer.doAfter(3, function()
                    hideTargetOverlay()
                end)
            else
                -- Couldn't resolve — show error briefly
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

    -- Hide the overlay once we start routing (target stays sticky in memory)
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

local function hasExplicitTarget(text)
    -- Check if the text starts with a routing verb (tell/ask/hey/go to/switch to/for/focus)
    local lower = text:lower()
    return lower:match("^tell%s") or lower:match("^ask%s") or lower:match("^send%s")
        or lower:match("^hey%s") or lower:match("^yo%s") or lower:match("^ping%s")
        or lower:match("^for%s") or lower:match("^message%s")
        or lower:match("^go%s+to%s") or lower:match("^switch%s+to%s") or lower:match("^focus%s")
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

    if hasExplicitTarget(text) then
        -- User said "tell X..." or "switch to X" — resolve new target
        voiceRouter.routing = false
        clearPendingTarget()
        resolveTarget(text)
    elseif voiceRouter.pendingTarget then
        -- Active target — route directly, stays sticky forever
        routeToTarget(voiceRouter.pendingTarget, text)
    else
        -- No target yet — resolve from speech
        voiceRouter.routing = false
        resolveTarget(text)
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

    elseif not rightOpt and voiceRouter.optDown then
        voiceRouter.optDown = false
        signalDaemon("USR2")
        hideMicIndicator()
        voiceRouter.pendingCount = voiceRouter.pendingCount + 1
        ensurePolling()
    end

    return false
end)

voiceRouter.tap:start()

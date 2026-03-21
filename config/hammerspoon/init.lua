
-- voice-claude v2 — hold Right Option to speak, text streams live into Claude Code
-- Single press: hold to record, release to submit
-- Routing: parser classifies target from text, falls back to last-active

local voiceRouter = {}
voiceRouter.optDown = false
voiceRouter.partialFile = os.getenv("HOME") .. "/.local/share/voice-claude/daemon-partial.json"
voiceRouter.resultsFile = os.getenv("HOME") .. "/.local/share/voice-claude/daemon-results.jsonl"
voiceRouter.pidFile = os.getenv("HOME") .. "/.local/share/voice-claude/listen-daemon.pid"
voiceRouter.pollTimer = nil
voiceRouter.lastResultLine = 0
voiceRouter.pendingCount = 0
voiceRouter.lastPartialText = ""
voiceRouter.chunkTimer = nil
voiceRouter.pendingTarget = nil  -- sticky target

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

-- Show a small text indicator (no image, just text)
local function showRecordingIndicator()
    hs.alert.show("🎤", nil, nil, 30)
end

local function hideRecordingIndicator()
    hs.alert.closeAll()
end

-- Poll daemon-partial.json for live streaming text
local function pollPartial()
    local f = io.open(voiceRouter.partialFile, "r")
    if not f then return end
    local raw = f:read("*a")
    f:close()
    if not raw or #raw == 0 then return end

    local ok, partial = pcall(hs.json.decode, raw)
    if not ok or not partial then return end

    -- Only update if text changed
    if partial.text and partial.text ~= voiceRouter.lastPartialText then
        voiceRouter.lastPartialText = partial.text
        if #partial.text > 0 then
            hs.alert.closeAll()
            local display = "🎤 " .. partial.text
            if voiceRouter.pendingTarget then
                display = "→ " .. voiceRouter.pendingTarget .. "\n🎤 " .. partial.text
            end
            hs.alert.show(display, nil, nil, 30)
        end
    end

    -- Route immediately when final text arrives
    if partial.final and partial.text and #partial.text > 0 and not voiceRouter.routed then
        voiceRouter.routed = true
        stopPartialPolling()
        hs.timer.doAfter(0.5, function() hs.alert.closeAll() end)
        routeFinalText(partial.text)
    end
end

local function startPartialPolling()
    if voiceRouter.chunkTimer then return end
    voiceRouter.lastPartialText = ""
    voiceRouter.chunkTimer = hs.timer.doEvery(0.2, function()
        pollPartial()
    end)
end

local function stopPartialPolling()
    if voiceRouter.chunkTimer then
        voiceRouter.chunkTimer:stop()
        voiceRouter.chunkTimer = nil
    end
end

-- Route the final text
local function routeFinalText(text)
    if not text or #text == 0 then return end

    local env = {
        PATH = os.getenv("HOME") .. "/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
        HOME = os.getenv("HOME"),
    }

    -- If we have a sticky target, classify first
    if voiceRouter.pendingTarget then
        -- Check for focus verbs (always switch)
        local lower = text:lower()
        if lower:match("^go%s+to%s") or lower:match("^switch%s+to%s") or lower:match("^focus%s") then
            -- Resolve new target
            local task = hs.task.new(
                os.getenv("HOME") .. "/.local/bin/voice-route",
                function(exitCode, stdout, stderr)
                    if exitCode == 0 and stdout then
                        local target = stdout:match("^%s*(.-)%s*$")
                        if target and #target > 0 then
                            voiceRouter.pendingTarget = target
                            hs.alert.show("→ " .. target, nil, nil, 2)
                        end
                    end
                end,
                {"--resolve", text}
            )
            task:setEnvironment(env)
            task:start()
            return
        end

        -- Classify: switch vs content
        local task = hs.task.new(
            os.getenv("HOME") .. "/.local/bin/voice-route",
            function(exitCode, stdout, stderr)
                if exitCode ~= 0 or not stdout then
                    -- Fallback: send as content
                    local t2 = hs.task.new(
                        os.getenv("HOME") .. "/.local/bin/voice-route",
                        function() end,
                        {"--target", voiceRouter.pendingTarget, "--text", text}
                    )
                    t2:setEnvironment(env)
                    t2:start()
                    return
                end

                local ok, result = pcall(hs.json.decode, stdout)
                if ok and result then
                    if result.action == "switch" and result.target then
                        voiceRouter.pendingTarget = result.target
                        hs.alert.show("→ " .. result.target, nil, nil, 2)
                        if result.text and #result.text > 0 then
                            local t2 = hs.task.new(
                                os.getenv("HOME") .. "/.local/bin/voice-route",
                                function() end,
                                {"--target", result.target, "--text", result.text}
                            )
                            t2:setEnvironment(env)
                            t2:start()
                        end
                    elseif result.action == "self" and result.text then
                        local t2 = hs.task.new(
                            os.getenv("HOME") .. "/.local/bin/voice-route",
                            function() end,
                            {"--target", voiceRouter.pendingTarget, "--text", result.text}
                        )
                        t2:setEnvironment(env)
                        t2:start()
                    else
                        local t2 = hs.task.new(
                            os.getenv("HOME") .. "/.local/bin/voice-route",
                            function() end,
                            {"--target", voiceRouter.pendingTarget, "--text", text}
                        )
                        t2:setEnvironment(env)
                        t2:start()
                    end
                end
            end,
            {"--classify", text, "--sticky-target", voiceRouter.pendingTarget}
        )
        task:setEnvironment(env)
        task:start()
    else
        -- No sticky target — use --text which handles routing + last-active
        local task = hs.task.new(
            os.getenv("HOME") .. "/.local/bin/voice-route",
            function(exitCode, stdout, stderr)
                -- Capture target from output
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

-- Consume final results after recording stops
local function consumeResults()
    local f = io.open(voiceRouter.resultsFile, "r")
    if not f then return end

    local lineNum = 0
    for line in f:lines() do
        lineNum = lineNum + 1
        if lineNum > voiceRouter.lastResultLine then
            local ok, result = pcall(hs.json.decode, line)
            if ok and result and result.text and #result.text > 0 then
                routeFinalText(result.text)
                voiceRouter.pendingCount = math.max(0, voiceRouter.pendingCount - 1)
            elseif ok then
                voiceRouter.pendingCount = math.max(0, voiceRouter.pendingCount - 1)
            end
            voiceRouter.lastResultLine = lineNum
        end
    end
    f:close()
end

local function ensurePolling()
    if voiceRouter.pollTimer then return end
    voiceRouter.pollTimer = hs.timer.doEvery(0.15, function()
        consumeResults()

        if voiceRouter.pendingCount <= 0 and not voiceRouter.optDown then
            -- Don't truncate — wait longer for daemon to write
            -- Only stop after we've actually consumed a result OR 5s passed
            if not voiceRouter.pollStartTime then
                voiceRouter.pollStartTime = hs.timer.secondsSinceEpoch()
            end
            local elapsed = hs.timer.secondsSinceEpoch() - (voiceRouter.pollStartTime or 0)
            if elapsed > 5 then
                local f = io.open(voiceRouter.resultsFile, "w")
                if f then f:close() end
                voiceRouter.lastResultLine = 0
                voiceRouter.pollStartTime = nil
                voiceRouter.pollTimer:stop()
                voiceRouter.pollTimer = nil
            end
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
        voiceRouter.routed = false
        signalDaemon("USR1")
        showRecordingIndicator()
        startPartialPolling()

    elseif not rightOpt and voiceRouter.optDown then
        voiceRouter.optDown = false
        signalDaemon("USR2")
        -- Don't stop partial polling yet — wait for final:true
        -- It will stop itself when it sees final and routes
        -- Safety: stop after 5s if no final arrives
        hs.timer.doAfter(5, function()
            stopPartialPolling()
            if not voiceRouter.routed then
                hs.alert.closeAll()
            end
        end)
    end

    return false
end)

voiceRouter.tap:start()
